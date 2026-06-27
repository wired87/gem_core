"""
gem.py

Prompt:
    "Split the image generation process and conversion process to two different routes.
     User confirms generation -> generate image -> render in result space (also include
     here a download button which downloads the single image) -> render file types
     check box section in the config space (allow single choice additionally to the
     generated image) incl second confirm button (this section is hidden till an image
     was generated) -> on confirm again load -> render generated file additionally to
     the prev generated image in the result space (no selected) -> download button now
     downloads both file in a zip"

    "uploaded images are not taken into account properly currently within the generation
     procedure. implement safe fix for that problem."

    When pasted/uploaded reference images are present, generation uses a Gemini image
    model with multimodal reference parts (GEM_IMAGE_MODEL_WITH_REFS). Text-only runs
    keep Imagen (GEM_IMAGE_MODEL / imagen-4.0-ultra-generate-001).

Prompt: *service currently unavailable on image generation — run pipeline, infer root cause,
implement robust multi-model fallback + clearer user-facing errors.*

Split workflow additions:
    - run_image_only_pipeline   – step 1: generate img.jpg + args.json into a caller-owned run dir.
    - run_conversion_pipeline   – step 2: produce exactly one derivative via converters/ package.

Prompt: *simplify file conversion logic — high quality png, svg and eps graphics.*
    Graphics (png/svg/eps) always via converters/; process_image handles mesh outputs only.

Prompt: *"take the size of the generated image into account ... optimize that process to
minimum size and maximized quality"* — convert paths call prepare_source_image_for_conversion()
before process_image (see conv_3d.py).

Prompt: *"created svg files are very pixelated"* — SVG convert uses prefer_high_res=True so
img.jpg keeps litho caps (LITHO_SVG_MAX_DIM) instead of the legacy 1024 mesh default.

The legacy `run_generation_pipeline` is preserved unchanged for the original `/api/process`
single-shot route so existing callers continue to work.
"""
import json
import sys

from conv_3d import prepare_source_image_for_conversion, process_image

# Prompt: fix Windows charmap crash on emoji pipeline logs during Create Media / generate image.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError, OSError):
    pass
import uuid
import tempfile
import shutil
import mimetypes
from types import SimpleNamespace
from typing import Optional
from google import genai
import dotenv
from google.genai import types
import argparse

import requests


import os
import time


from converters import (
    ConvertContext,
    run_conversion,
)

dotenv.load_dotenv()


def _validate_non_flat_stl(stl_path: str, min_z_span: float = 0.05):
    """
    Validate that STL has real 3D relief (non-flat geometry).
    Delegates to conv_3d._validate_mesh_3d when available so the
    same white=high / black=low correlation check applies.
    """
    if not os.path.isfile(stl_path):
        raise RuntimeError(f"STL fehlt: {stl_path}")

    try:
        from core.conv_3d import _validate_mesh_3d
        result = _validate_mesh_3d(stl_path, label="STL[gem]", min_z_span=min_z_span)
        if not result["ok"]:
            issues = "; ".join(result["issues"])
            raise RuntimeError(f"STL validation failed: {issues}")
        return
    except ImportError:
        pass

    try:
        import trimesh
        mesh = trimesh.load_mesh(stl_path)

        if mesh is None or getattr(mesh, "vertices", None) is None or len(mesh.vertices) == 0:
            raise RuntimeError("STL ist leer oder konnte nicht geladen werden")

        z_values = mesh.vertices[:, 2]
        z_span = float(z_values.max() - z_values.min())

        if z_span <= float(min_z_span):
            raise RuntimeError(
                f"STL ist flach (z_span={z_span:.6f}, min_required={min_z_span:.6f})"
            )
    except ImportError:
        # Fallback when trimesh is unavailable at runtime.
        min_size_bytes = 1024
        size_bytes = os.path.getsize(stl_path)
        if size_bytes < min_size_bytes:
            raise RuntimeError(
                f"STL wirkt ungueltig/zu klein ({size_bytes} bytes); trimesh nicht verfuegbar"
            )



# ---------- 1. QUERY TRANSFORMATION ----------
def transform_query(full_prompt: str, gem_api_key) -> str:
    """
    Nimmt eine statische Instruktion und einen Basis-Text (Preprint),
    um über das Textmodell einen perfekten Bild-Prompt zu generieren.
    """
    # AI: human-readable progress print (no bracket prefixes)
    print("Optimizing the prompt with Gemini.")
    try:
        client = genai.Client(api_key=gem_api_key or os.getenv("GEM_API_KEY"))
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1
            )
        )

        optimized_prompt = response.text.strip()
        # AI: keep output short; do not dump the full prompt
        print("Prompt optimization finished.")
        return f"{optimized_prompt}"

    except Exception as e:
        # AI: short error print without bracket prefix
        print(f"Prompt optimization failed: {e}")
        return ""



def list_available_models(gem_api_key):
    # AI: short, plain status print
    print("Fetching available Gemini models.")
    client = genai.Client(api_key=gem_api_key or os.getenv("GEM_API_KEY"))
    try:
        # client.models.list() gibt einen Iterator aller verfügbaren Modelle zurück
        models = client.models.list()

        print("--- ALLE VERFÜGBAREN MODELLE ---")
        for m in models:
            # Zeigt den Namen des Modells an (z.B. 'models/gemini-2.5-pro')
            print(f"Name: {m.name}")

            # Zeigt an, was das Modell kann (z.B. 'generateContent', 'generateImages')
            if m.supported_actions:
                print(f"  -> Aktionen: {', '.join(m.supported_actions)}")
            print("-" * 30)

    except Exception as e:
        # AI: short error print without bracket prefix
        print(f"Failed to fetch available models: {e}")

def get_prompt(images, theme, bg_texture, math_rule, product_name, typo_style, color_palette, tags):
    # --- Modular Variables ---
    """THEME_DESCRIPTION = "Mathematical and physical futuristic"
    BACKGROUND_TEXTURE = "sharp"
    GEOMETRIC_MATHEMATICAL_RULE = "golden ratio proportions"
    PRODUCT_NAME = ""
    TYPOGRAPHY_STYLE = "futuristic"
    COLOR_PALETTE = "black and white"
    TAGS = "A dark luxury cyberpunk lighter-cover with glowing orange neon elements and elegant typography."
    INPUT_DIR = "input"
    IMAGES = [
        Image.open(os.path.join(INPUT_DIR, file))
        for file in os.listdir(INPUT_DIR)
        if file.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]"""

    def include_text():
        if len(product_name) > 0:
            return f"""
            The product name '{product_name}' is integrated into the layout using {typo_style}.
            """
        else:
            return "Do not include any title or text on the image!!!"

    static_prompt = f"""
    Generate a premium, flat graphic design for a square adhesive wrap (1:1 aspect ratio) based on the following strict parameters:

    * **Core Medium:** A flat, strictly 2D graphic design file. It is a full-bleed, edge-to-edge layout, strictly restricted to pure artwork.
    * **CRITICAL INSTRUCTION:** Do NOT draw any physical objects, hardware, or mockups. Draw ONLY the pure 2D print pattern itself.
    * **Perspective:** Orthographic, top-down view with absolutely no 3D perspective, no shadows, and no curved surfaces.
    * **Layout Exclusions:** No external text measurements, no dimension lines, no white borders, and no realistic product mockup backgrounds.
    * **Theme & Geometry:** The design theme features '{theme}' on a background texture of '{bg_texture}'. 
    Patterns and geometric structures are strictly derived from precise mathematical algorithms and physical forms, 
    specifically {math_rule}.
    * **Aesthetics:** High contrast lighting optimized for flat graphic print, with vector-style crispness.
    * **Colors:** The color palette is strictly: {color_palette}.
    * **Banned Effects:** No atmospheric smoke, bokeh, reflections, or out-of-focus areas.
    * **Typography:** {include_text()}
    * **Style tags / mood:** {tags.strip() or 'premium collectible cover art'}

    Rules: 
    STRICT TECHNICAL LAYOUT RULES

    - Full-bleed composition only.
    - Artwork must extend to all canvas boundaries.
    - No outer margins, padding, whitespace, safe zones, gutters, borders, framing elements, crop marks, bleed marks, registration marks, rulers, guides, measurement annotations, engineering callouts, dimension lines, arrows, labels, captions, technical notes, legends, scale indicators, coordinates, drafting overlays, reference grids, construction marks, specification text, typography, symbols, watermarks, signatures, logos, or metadata.

    - Geometry must occupy the entire available canvas area with visually balanced edge coverage.
    - No unused negative space around the perimeter.
    - No inset artwork.
    - No centered floating object surrounded by empty background.
    - Primary design elements must reach or visually interact with all four canvas edges.

    - Output must appear as final production artwork rather than a concept sketch, blueprint, mockup, presentation board, technical drawing, engineering diagram, CAD rendering, architectural sheet, manufacturing specification, or print preview.

    - Exclude all dimensional information including width, height, scale, proportions, measurements, units, numerical annotations, arrows, extension lines, and reference markers.

    - No visible paper background.
    - No framing rectangle around the design.
    - No drop shadows suggesting a separate object placed on a page.

    - Render only the finished design itself.
    - Edge-to-edge coverage required.

    """

    reference_block = ""
    ref_labels = _reference_image_labels(images)
    if ref_labels:
        joined = ", ".join(ref_labels)
        reference_block = f"""
    --- UPLOADED REFERENCE IMAGES ---
    The user supplied {len(ref_labels)} reference image(s): {joined}.
    The attached image(s) are style / composition / palette / motif references.
    Adapt them into a NEW square flat print design — do NOT output a product mockup photo.
    Preserve relevant visual language (geometry, contrast, typography feel, color relationships).
    """

    full_prompt = f"""
    {static_prompt.strip()}
    {reference_block.strip()}

    --- PREPRINT TEXT ---
    {math_rule}
    """

    return full_prompt


def _max_reference_images() -> int:
    raw = (os.getenv("MAX_PASTED_IMAGES") or "8").strip()
    try:
        return max(1, min(14, int(raw)))
    except ValueError:
        return 8


def _reference_image_labels(images) -> list:
    labels = []
    for idx, item in enumerate(images or [], start=1):
        if not isinstance(item, dict):
            continue
        if item.get("source") == "file":
            labels.append(os.path.basename(str(item.get("path") or f"reference_{idx}")))
        elif item.get("source") == "url":
            labels.append(str(item.get("url") or f"reference_url_{idx}"))
    return labels


def _guess_image_mime(path: str, default: str = "image/jpeg") -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime if mime and mime.startswith("image/") else default


def _build_reference_parts(images):
    """Build Gemini multimodal parts for uploaded reference images."""
    parts = []
    labels = []
    max_refs = _max_reference_images()
    for item in images or []:
        if len(parts) >= max_refs:
            break
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "")
        try:
            if source == "file":
                path = str(item.get("path") or "").strip()
                if not path or not os.path.isfile(path):
                    continue
                with open(path, "rb") as fh:
                    data = fh.read()
                if not data:
                    continue
                parts.append(types.Part.from_bytes(data=data, mime_type=_guess_image_mime(path)))
                labels.append(os.path.basename(path))
            elif source == "url":
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                parts.append(types.Part.from_uri(file_uri=url, mime_type="image/jpeg"))
                labels.append(url)
        except Exception as ref_error:
            # AI: short warning print without bracket prefix
            print(f"Skipped a reference image: {ref_error}")
    return parts, labels


def _extract_image_bytes_from_content_response(response) -> list:
    """Pull raw JPEG/PNG bytes from a Gemini generate_content image response."""
    collected = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in (getattr(content, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                collected.append(inline.data)
                continue
            if hasattr(part, "as_image"):
                try:
                    pil_img = part.as_image()
                    from io import BytesIO
                    buf = BytesIO()
                    pil_img.save(buf, format="JPEG", quality=92)
                    collected.append(buf.getvalue())
                except Exception:
                    pass
    return collected


def _wrap_generated_image_bytes(image_bytes_list):
    """Adapt Gemini bytes into the same shape Imagen returns (generated_images[].image.image_bytes)."""
    wrapped = []
    for raw in image_bytes_list or []:
        if not raw:
            continue
        wrapped.append(SimpleNamespace(image=SimpleNamespace(image_bytes=raw)))
    return SimpleNamespace(generated_images=wrapped)


def _genai_client(gem_api_key=None):
    """Shared Gemini/Imagen client for cover generation."""
    return genai.Client(api_key=gem_api_key or os.getenv("GEM_API_KEY"))


def _imagen_model_chain() -> list:
    """Primary Imagen model plus ordered fallbacks (env GEM_IMAGE_MODEL_FALLBACKS)."""
    primary = (os.getenv("GEM_IMAGE_MODEL") or "imagen-4.0-ultra-generate-001").strip()
    raw_fallbacks = (os.getenv("GEM_IMAGE_MODEL_FALLBACKS") or "").strip()
    if raw_fallbacks:
        fallbacks = [m.strip() for m in raw_fallbacks.split(",") if m.strip()]
    else:
        # Prompt: ultra tier often 503/429 — fast + standard Imagen tiers recover quickly.
        fallbacks = ["imagen-4.0-fast-generate-001", "imagen-4.0-generate-001"]
    chain: list = []
    for model_id in [primary, *fallbacks]:
        if model_id and model_id not in chain:
            chain.append(model_id)
    return chain


def _gemini_text_fallback_model() -> str:
    """Last-resort text-to-image model when every Imagen tier fails."""
    return (os.getenv("GEM_IMAGE_TEXT_FALLBACK_MODEL") or "gemini-2.5-flash-image").strip()


def _generation_error_code(exc: Exception) -> Optional[int]:
    """Best-effort HTTP/gRPC status code from google.genai errors."""
    for attr in ("status_code", "code"):
        raw = getattr(exc, attr, None)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    text = str(exc).upper()
    for code in (503, 429, 404, 400, 401, 403):
        if str(code) in text:
            return code
    return None


def _is_transient_generation_error(exc: Exception) -> bool:
    """True for overload / capacity errors where another model or retry may succeed."""
    code = _generation_error_code(exc)
    if code in (429, 503, 504):
        return True
    text = str(exc).upper()
    return any(
        token in text
        for token in (
            "UNAVAILABLE",
            "RESOURCE_EXHAUSTED",
            "OVERLOADED",
            "DEADLINE",
            "TIMEOUT",
            "TRY AGAIN",
            "CURRENTLY UNAVAILABLE",
        )
    )


def _is_permanent_model_error(exc: Exception) -> bool:
    """404/400 on a model id — skip retries and try the next model in the chain."""
    code = _generation_error_code(exc)
    if code in (404, 400):
        return True
    text = str(exc).upper()
    return "NOT_FOUND" in text or "IS NOT FOUND" in text


def user_facing_generation_error(exc: Exception) -> str:
    """Map provider failures to actionable UI copy (no raw stack traces)."""
    if _is_transient_generation_error(exc):
        return (
            "Google image generation is temporarily overloaded. "
            "The server tried multiple models — please wait a moment and try again."
        )
    text = str(exc).strip()
    if "API key" in text or _generation_error_code(exc) == 401:
        return "Gemini API key is missing or invalid. Check GEM_API_KEY in your server configuration."
    if len(text) > 240:
        text = text[:237] + "..."
    return text or "Image generation failed. Please try again."


def _generate_cover_image_imagen_once(image_prompt: str, model: str, gem_api_key=None):
    """Single Imagen generate_images call for *model*."""
    client = _genai_client(gem_api_key)
    return client.models.generate_images(
        model=model,
        prompt=image_prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            output_mime_type="image/jpeg",
            aspect_ratio="1:1",
        ),
    )


def _generate_cover_image_imagen(image_prompt: str, gem_api_key=None):
    """Imagen with ordered model fallbacks and light per-model retries on transient errors."""
    per_model_retries = max(1, int(os.getenv("GEM_IMAGE_RETRY_PER_MODEL") or "1"))
    retry_delays = (2, 5, 10)
    collected_errors: list = []

    for model in _imagen_model_chain():
        for attempt in range(per_model_retries):
            try:
                # AI: short progress print without bracket prefix
                print(f"Generating image (attempt {attempt + 1}/{per_model_retries}).")
                return _generate_cover_image_imagen_once(image_prompt, model, gem_api_key)
            except Exception as exc:
                collected_errors.append(f"{model}: {exc}")
                # AI: short warning print without bracket prefix
                print(f"Image generation attempt failed: {exc}")
                if _is_permanent_model_error(exc):
                    break
                if not _is_transient_generation_error(exc):
                    raise
                if attempt + 1 < per_model_retries:
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                    # AI: short retry print without bracket prefix
                    print(f"Temporary error. Retrying in {delay}s.")
                    time.sleep(delay)
        # AI: short fallback print without bracket prefix
        print("Primary image model exhausted. Trying a fallback.")

    raise RuntimeError(
        "All Imagen models failed: " + " | ".join(collected_errors[-3:])
    )


def _gemini_image_generate_config():
    """Developer-API-safe image config (output_mime_type breaks on non-Enterprise keys)."""
    return types.GenerateContentConfig(response_modalities=["IMAGE"])


def _generate_cover_image_gemini_text(image_prompt: str, gem_api_key=None):
    """Text-only fallback via Gemini native image output when Imagen tiers are down."""
    model = _gemini_text_fallback_model()
    client = _genai_client(gem_api_key)
    # AI: short fallback print without bracket prefix
    print("Trying Gemini text-only fallback.")
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=image_prompt)])],
        config=_gemini_image_generate_config(),
    )
    image_bytes_list = _extract_image_bytes_from_content_response(response)
    if not image_bytes_list:
        raise RuntimeError(f"Gemini fallback model={model} returned no image bytes")
    return _wrap_generated_image_bytes(image_bytes_list)


def _generate_cover_image_with_references(image_prompt: str, images, gem_api_key=None):
    reference_parts, ref_labels = _build_reference_parts(images)
    if not reference_parts:
        raise ValueError("Reference image generation requested but no readable reference parts were built")

    client = _genai_client(gem_api_key)
    model = (os.getenv("GEM_IMAGE_MODEL_WITH_REFS") or "gemini-2.5-flash-image").strip()
    # AI: avoid printing large lists; keep it short
    print(f"Generating with reference images (count: {len(reference_parts)}).")

    content_parts = list(reference_parts)
    content_parts.append(types.Part.from_text(text=image_prompt))
    # Prompt: Developer API rejects output_mime_type on ImageConfig — use response_modalities only.
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=content_parts)],
        config=_gemini_image_generate_config(),
    )
    image_bytes_list = _extract_image_bytes_from_content_response(response)
    if not image_bytes_list:
        raise RuntimeError("Reference-aware generation returned no image bytes")
    return _wrap_generated_image_bytes(image_bytes_list)


def _persist_reference_inputs(images, output_dir: str) -> list:
    """Copy uploaded reference files into the run dir so they appear in result artifacts."""
    if not output_dir or not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    saved = []
    for idx, item in enumerate(images or [], start=1):
        if idx > _max_reference_images():
            break
        if not isinstance(item, dict) or item.get("source") != "file":
            continue
        src = str(item.get("path") or "").strip()
        if not src or not os.path.isfile(src):
            continue
        dest_name = f"ref_input_{idx:02d}_{os.path.basename(src)}"
        dest = os.path.join(output_dir, dest_name)
        shutil.copy2(src, dest)
        saved.append(dest)
    if saved:
        # AI: short persistence print without bracket prefix
        print(f"Saved {len(saved)} reference input(s) for this run.")
    return saved


def generate_cover_image(image_prompt: str, gem_api_key=None, images=None):
    """
    Generate cover art. Uses Gemini reference-image model when uploads are present,
    otherwise Imagen text-to-image with multi-model + Gemini text fallback.
    """
    # AI: short progress print without bracket prefix
    print("Generating the image.")
    has_refs = bool(_reference_image_labels(images))
    try:
        if has_refs:
            try:
                return _generate_cover_image_with_references(image_prompt, images, gem_api_key)
            except Exception as ref_error:
                # AI: short warning print without bracket prefix
                print(f"Reference-aware generation failed. Falling back: {ref_error}")
        try:
            return _generate_cover_image_imagen(image_prompt, gem_api_key)
        except Exception as imagen_error:
            # AI: short warning print without bracket prefix
            print(f"Imagen chain failed. Trying fallback: {imagen_error}")
            return _generate_cover_image_gemini_text(image_prompt, gem_api_key)
    except Exception as e:
        # AI: short error print without bracket prefix
        print(f"Image generation failed: {e}")
        raise




# ---------- 1. DIE NÄCHSTE FUNKTION (Pipeline) ----------
def run_generation_pipeline(
        images,
        theme,
        bg_texture,
        math_rule,
        product_name,
        typo_style,
        color_palette,
        tags,
        output_dir,
        args,
        gem_api_key,
        args_payload=None,
        height=600,
        width=600,
        generate_svg=True,
        generate_stl=True,
        generate_html=True,
        generate_animation=True,
        generate_glb=False,
        generate_obj=False,
        generate_3mf=False,
        generate_png=False,
):
    """
    Diese Funktion nimmt alle geöffneten Bilder und Argumente entgegen und
    könnte nun deinen Prompt bauen und an die Google GenAI API schicken.
    """
    print("\n🚀 --- PIPELINE GESTARTET ---")
    print(f"📦 Geladene Bilder:   {len(images)}")
    print(f"🎨 Theme:             {theme}")
    print(f"🖼️ Background:        {bg_texture}")
    print(f"📐 Math Rule:         {math_rule}")
    print(f"🏷️ Product Name:      '{product_name}'")
    print(f"🔤 Typography:        {typo_style}")
    print(f"🎨 Colors:            {color_palette}")
    print(f"🔖 Tags:              {tags}")
    print("----------------------------\n")


    ### VALIDATE
    if os.getenv("SHOW_AVAILABLE_MODELS", None) is not None:
        list_available_models()

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="lighter0_run_")
        print("no output dir given -> using temp store", output_dir)
    else:
        output_dir = tempfile.mkdtemp(prefix="lighter0_run_")
        print("output dir ignored -> using temp store", output_dir)
    ###

    prompt = get_prompt(images, theme, bg_texture, math_rule, product_name, typo_style, color_palette, tags)
    # GENERATE
    gen_id = uuid.uuid4()

    # 1. Erstelle den spezifischen Unterordner für DIESEN Durchlauf
    run_dir = os.path.join(output_dir, str(gen_id))
    os.makedirs(run_dir, exist_ok=True)

    # Persist uploaded references beside generated img.jpg for result-space visibility.
    _persist_reference_inputs(images, run_dir)

    # 2. Definiere die finalen Speicherpfade
    image_save_path = os.path.join(run_dir, "img.jpg")
    json_save_path = os.path.join(run_dir, "args.json")

    result = generate_cover_image(prompt, gem_api_key, images=images)

    # --- SAVE IMAGE ---
    for generated_image in result.generated_images:
        image_bytes = generated_image.image.image_bytes
        with open(image_save_path, "wb") as image_file:
            image_file.write(image_bytes)
        # AI: short success print without bracket prefix
        print("Image saved.")

    args_dict = args_payload if isinstance(args_payload, dict) else vars(args)

    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)

    # Normalize img.jpg once, then route graphics vs mesh through dedicated modules.
    needs_hires = bool(generate_svg or generate_stl or generate_glb or generate_obj or generate_3mf or generate_png)
    prepare_source_image_for_conversion(image_save_path, run_dir=run_dir, prefer_high_res=needs_hires)

    ctx = ConvertContext(run_dir=run_dir, source_path=image_save_path)
    run_conversion("eps", ctx)
    if generate_png:
        run_conversion("png", ctx)
    if generate_svg:
        run_conversion("svg", ctx)

    needs_mesh = any([
        generate_stl, generate_html, generate_animation,
        generate_glb, generate_obj, generate_3mf,
    ])
    if needs_mesh:
        process_result = process_image(
            run_dir,
            image_save_path,
            generate_stl=generate_stl,
            generate_html=generate_html,
            generate_animation=generate_animation,
            generate_glb=generate_glb,
            generate_obj=generate_obj,
            generate_3mf=generate_3mf,
        )
    else:
        process_result = {"stl_status": "skipped", "stl_info": "skipped (graphics-only)"}

    # Enforce required artifact set for frontend integration.
    # Only validate files that were requested.
    required_files = ["img.jpg", "args.json", "vec.eps"]
    if generate_svg:
        required_files.append("out.svg")
    if generate_html:
        required_files.append("out.html")
    if generate_animation:
        required_files += ["animated.svg", "brainmaster.json"]
    if generate_stl:
        required_files.append("out.stl")
    if generate_png:
        required_files.append("out.png")
    # GLB/OBJ/3MF are best-effort; missing them is only a warning not a hard failure,
    # because trimesh format support may be unavailable at runtime.
    optional_files = []
    if generate_glb:
        optional_files.append("out.glb")
    if generate_obj:
        optional_files.append("out.obj")
    if generate_3mf:
        optional_files.append("out.3mf")
    missing_files = [
        name for name in required_files
        if not os.path.isfile(os.path.join(run_dir, name))
    ]

    stl_status = "unknown"
    if isinstance(process_result, dict):
        stl_status = str(process_result.get("stl_status", "unknown"))

    if missing_files:
        missing_text = ", ".join(missing_files)
        raise RuntimeError(
            f"Pipeline unvollständig. Fehlende Artefakte: {missing_text}. "
            f"STL-Status: {stl_status}"
        )

    # Warn (but don't fail) for optional compact mesh formats.
    missing_optional = [
        name for name in optional_files
        if not os.path.isfile(os.path.join(run_dir, name))
    ]
    if missing_optional:
        # AI: keep output short; no long file lists
        print("Some optional mesh exports were not created.")

    if generate_stl:
        stl_path = os.path.join(run_dir, "out.stl")
        _validate_non_flat_stl(stl_path)

    # AI: short persistence print without bracket prefix
    print("Saved run parameters.")

    generated_files = []
    try:
        for name in sorted(os.listdir(run_dir)):
            path = os.path.join(run_dir, name)
            if os.path.isfile(path):
                generated_files.append(path)
    except Exception as e:
        # AI: short warning print without bracket prefix
        print(f"Could not list generated files: {e}")

    return {
        "run_dir": run_dir,
        "gen_id": str(gen_id),
        "generated_files": generated_files,
        "process_result": process_result,
        "process_payload": args_dict,
    }


# ============================================================================
# SPLIT WORKFLOW PIPELINES
# ----------------------------------------------------------------------------
# Two-step user flow used by the /api/process/image + /api/process/convert
# endpoints: first call generates only the cover image (img.jpg + args.json),
# the second call adds exactly one derived format (svg / stl / html / animation
# / glb / obj / 3mf) using the *same* run directory so the user is not charged
# again. Both pipelines are explicit about their output_dir and never silently
# replace it with a system temp directory (unlike `run_generation_pipeline`).
# ============================================================================


# Step 1 of the split workflow: generate only the AI cover image into the caller-provided run directory.
def run_image_only_pipeline(
        images,
        theme,
        bg_texture,
        math_rule,
        product_name,
        typo_style,
        color_palette,
        tags,
        output_dir,
        args,
        gem_api_key,
        args_payload=None,
        height=600,
        width=600,
):
    """
    Image-only stage of the split workflow.

    Generates `img.jpg` (the AI cover image) and `args.json` (the request payload echo)
    inside `output_dir`. Does NOT run vector conversion or 3D post-processing — those
    happen in `run_conversion_pipeline` once the user picks a single derivative format.

    `output_dir` is required and must already be writable. Caller is responsible for
    cleanup; this stage intentionally leaves the directory in place so the second
    step can reuse the generated image.
    """
    print("\n🚀 --- IMAGE-ONLY PIPELINE GESTARTET ---")
    print(f"📦 Geladene Bilder:   {len(images)}")
    print(f"🎨 Theme:             {theme}")
    print(f"🖼️ Background:        {bg_texture}")
    print(f"📐 Math Rule:         {math_rule}")
    print(f"🏷️ Product Name:      '{product_name}'")
    print(f"🔤 Typography:        {typo_style}")
    print(f"🎨 Colors:            {color_palette}")
    print(f"🔖 Tags:              {tags}")
    print(f"📁 Output dir:        {output_dir}")
    print("----------------------------\n")

    if not output_dir:
        raise ValueError("run_image_only_pipeline requires an explicit output_dir")

    if os.getenv("SHOW_AVAILABLE_MODELS", None) is not None:
        list_available_models()

    os.makedirs(output_dir, exist_ok=True)

    # Keep uploaded references in the run dir for result-space listing / download.
    _persist_reference_inputs(images, output_dir)

    prompt = get_prompt(images, theme, bg_texture, math_rule, product_name, typo_style, color_palette, tags)

    image_save_path = os.path.join(output_dir, "img.jpg")
    json_save_path = os.path.join(output_dir, "args.json")

    result = generate_cover_image(prompt, gem_api_key, images=images)

    image_written = False
    for generated_image in result.generated_images:
        image_bytes = generated_image.image.image_bytes
        with open(image_save_path, "wb") as image_file:
            image_file.write(image_bytes)
        image_written = True
        # AI: short success print without bracket prefix
        print("Image saved.")
        # Split workflow only needs the first generated image; conversion runs against this single asset.
        break

    if not image_written:
        raise RuntimeError("Image generation returned no images")

    args_dict = args_payload if isinstance(args_payload, dict) else vars(args)
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)

    if not os.path.isfile(image_save_path):
        raise RuntimeError(f"Image-only pipeline expected to produce {image_save_path}")

    return {
        "run_dir": output_dir,
        "image_path": image_save_path,
        "process_payload": args_dict,
    }


# ---------- 2. BILDER LADEN (Web, Datei, Ordner) ----------
def load_image_source(source_path: str):
    """
    Erkennt automatisch, ob der Pfad eine URL, ein lokaler Ordner oder eine Datei ist,
    und gibt speicherschonende Bild-Referenzen für den Prompt zurück.
    """
    loaded_images = []

    # Fall A: Es ist eine Web-URL
    if source_path.startswith("http://") or source_path.startswith("https://"):
        # AI: short, plain status print
        print("Loading an image from the web.")
        try:
            response = requests.get(source_path, stream=True, timeout=10)
            response.raise_for_status()  # Wirft Fehler bei 404 etc.
            content_type = str(response.headers.get("content-type") or "").lower()
            if content_type.startswith("image/"):
                loaded_images.append({"source": "url", "url": source_path})
            else:
                # AI: short warning print without bracket prefix
                print("The URL did not look like an image.")
            response.close()
        except Exception as e:
            # AI: short error print without bracket prefix
            print(f"Failed to download the image: {e}")

    # Fall B: Es ist ein lokaler Ordner
    elif os.path.isdir(source_path):
        # AI: short, plain status print
        print("Loading images from a folder.")
        for file in os.listdir(source_path):
            path = os.path.join(source_path, file)
            # Nur echte Dateien öffnen, die Endungen von Bildern haben
            if os.path.isfile(path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                try:
                    loaded_images.append({"source": "file", "path": path})
                except Exception as e:
                    # AI: short error print without bracket prefix
                    print(f"Failed to load one image: {e}")

    # Fall C: Es ist eine einzelne lokale Datei
    elif os.path.isfile(source_path):
        # AI: short, plain status print
        print("Loading a local image file.")
        try:
            if source_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                loaded_images.append({"source": "file", "path": source_path})
            else:
                # AI: short warning print without bracket prefix
                print("The file type is not a supported image format.")
        except Exception as e:
            # AI: short error print without bracket prefix
            print(f"Failed to load the local file: {e}")

    else:
        # AI: short warning print without bracket prefix
        print("Input path not found. Continuing without images.")


    return loaded_images

def print_welcome_screen():
    # ANSI Farb-Codes für das Terminal
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # ASCII Art Generator (Schriftart: "Slant")
    logo = rf"""
{CYAN}   _____                 __               {MAGENTA}  ___         __ 
{CYAN}  / ___/___  ____ _____/ /__  ___________{MAGENTA} /   |  _____/ /_
{CYAN}  \__ \/ _ \/ __ `/ __  / _ \/ ___/ ___/{MAGENTA} / /| | / ___/ __/
{YELLOW} ___/ /  __/ /_/ / /_/ /  __(__  |__  ) {YELLOW}/ ___ |/ /  / /_  
{YELLOW}/____/\___/\__,_/\__,_/\___/____/____/ {YELLOW}/_/  |_/_/   \__/  
{RESET}
    """
    print(logo)
    print(f"{BOLD} Seamless Cover-Art Generator CLI v1.0{RESET}")
    print(f" Powered by AI | {MAGENTA}Ready to create...{RESET}")
    print("-" * 55 + "\n")


def ask_user(frage: str, default_wert: str) -> str:
    print(f"❓ {frage}")

    # Wandle None in einen leeren String um, damit es keine TypeErrors gibt
    safe_default = "" if default_wert is None else str(default_wert)

    try:
        # Optional dependency: keep CLI UX when available, but never break API startup.
        from prompt_toolkit import prompt
        response = prompt("   > ", default=safe_default).strip()

    except Exception as e:
        #print(f"Cant open prompt toolkit. Falling back to default input... (Err, {e})")
        try:
            response = input(f"   [Default: {safe_default}]: ").strip()
            if not response:
                response = safe_default
        except Exception as e_fallback:
            print(f"   [!] Eingabe nicht möglich ({e_fallback}). Nutze Standardwert.", safe_default)
            response = safe_default

    if response == "exit":
        main()
    else:
        return response



def convert_to_vector_eps(
        input_path: str,
        output_eps_path: str,
        black_threshold: int = 55,
):
    """
    Legacy wrapper — delegates to converters/eps.py (rembg → vtracer → EPS).

    Prompt: *simplify file conversion logic — high quality png, svg and eps graphics.*
    """
    from converters.eps import generate_eps_from_image

    run_dir = os.path.dirname(output_eps_path) or None
    return generate_eps_from_image(
        input_path,
        output_eps_path,
        run_dir=run_dir,
        black_threshold=black_threshold,
    )


# ---------- 3. CLI SETUP ----------
def main():
    # PRINT WELCOME
    print_welcome_screen()
    # AI: short system print without bracket prefix
    print("Starting the pipeline.")

    # ArgumentParser generiert automatisch das -h / --help Menü
    parser = argparse.ArgumentParser(
        description="Generiere Seamless Cover-Artworks aus Bildern und Text-Parametern.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Zeigt die Defaults im Help-Text an
    )

    # Input Path (Kann URL, Ordner oder Datei sein)
    parser.add_argument("-i", "--input", default="input",
                        help="Pfad zum lokalen Ordner, zur Datei oder eine direkte Web-Bild-URL (z.B. https://example.com/img.jpg)")

    # Design Parameter
    parser.add_argument("--theme", default="Mathematical and physical futuristic",
                        help="Das Design-Thema (z.B. 'Cyberpunk neon city')")
    parser.add_argument("--bg_texture", default="sharp",
                        help="Hintergrundtextur (z.B. 'matte dark metal')")
    parser.add_argument("--math", default="golden ratio proportions",
                        help="Mathematische Geometrie-Regel (z.B. 'Fibonacci spirals')")
    parser.add_argument("--name", default="",
                        help="Der Produktname (Leer lassen für kein Text)")
    parser.add_argument("--typo", default="futuristic",
                        help="Der Typografie-Stil (z.B. 'bold angular modern')")
    parser.add_argument("--colors", default="black and white",
                        help="Farbpalette (z.B. 'neon orange and deep black')")
    parser.add_argument("--tags",
                        default="A colorful luxury cyberpunk lighter-cover with glowing orange neon elements and elegant typography.",
                        help="Zusätzliche Tags als Basis für das Design")
    parser.add_argument("--output_dir",
                        default="output",
                        help="Folder/Dir to save the genrated content to")
    parser.add_argument("--height",
                            default="600",
                            help="Höhe in mm (nur den Zahlen Wert)")
    parser.add_argument("--width",
                            default="600",
                            help="Breite in mm (nur den Zahlen Wert)")

    # Liest die Argumente aus der Kommandozeile
    args = parser.parse_args()

    # len(sys.argv) == 1 bedeutet: Der User hat "python main.py" ohne --parameter getippt
    if len(sys.argv) == 1:
        gem_api_key = input("❓ GEMINI_API_KEY (required): ").strip()
        if gem_api_key is None or len(gem_api_key) == 0:
            gem_api_key = os.getenv("GEM_API_KEY")
        if gem_api_key is None or len(gem_api_key) == 0:
            print("GEM KEY IS NOT ALLOWED TO BE EMPTY -> exit")
            main()
        json_input = input("❓ Optional: Pfad zu einer cfg.json Datei zum prefill der argumente - kann angepasst werden. (Leer lassen zum Überspringen): ").strip()

        # Bereinigt den Pfad (entfernt versehentliche Anführungszeichen beim Copy-Paste und löst "~" auf)
        json_path = os.path.expanduser(json_input.strip('"').strip("'"))

        if json_path:
            if os.path.isfile(json_path) and json_path.lower().endswith('.json'):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)

                    print(f"✅ JSON erfolgreich geladen: {json_path}")

                    # Mappe JSON-Keys auf args (nur wenn das Argument im Parser existiert)
                    args_dict = vars(args)
                    for key, value in config_data.items():
                        if key in args_dict:
                            setattr(args, key, value)
                            print(f"   -> Übernehme '{key}': {value}")

                except Exception as e:
                    print(f"❌ Fehler beim Lesen der JSON-Datei: {e}")
            else:
                print(f"⚠️ Datei nicht gefunden oder ungültig: '{json_path}'. Nutze Standardwerte.")

        print("\n--- Parameter-Setup ---")

        print("\n⚙️ Interaktiver Modus gestartet! (Drücke einfach ENTER für die Standardwerte)\n")
        args.input = ask_user("Wo liegen die Bilder? (Ordner, Datei oder URL)", args.input)
        args.tags = ask_user("Welche Tags beschreiben das Cover?", args.tags)
        args.theme = ask_user("Was ist das grundlegende Thema?", args.theme)
        args.bg_texture = ask_user("Wie soll die Hintergrundtextur sein?", args.bg_texture)
        args.math = ask_user("Gibt es eine geometrische Regel?", args.math)
        args.name = ask_user("Wie lautet der Produktname? (Leer für keinen Text)", args.name)
        args.typo = ask_user("Welcher Typografie-Stil soll genutzt werden?", args.typo)
        args.colors = ask_user("Wie lautet die Farbpalette?", args.colors)
        args.tags = ask_user("Zusätzliche Freitext-Beschreibung?", args.tags)
        args.output_dir = ask_user("In welcher Dir soll das resultet gespeichert werden?", args.output_dir)

        args.height = ask_user("Höhe in mm (nur den Zahlen Wert):", args.height)
        args.width = ask_user("Breite in mm (nur den Zahlen Wert):", args.width)
        print("\n" + "=" * 40 + "\n")

    try:
        args.height = int(args.height)
        args.width = int(args.width)
    except Exception as e:
        raise ValueError(f"Width and height not numerical: {e} \n->restart")
        main()

    # 1. Bilder laden
    images = load_image_source(args.input)

    # 2. Alles an die nächste Funktion übergeben
    run_generation_pipeline(
        images=images,
        theme=args.theme,
        bg_texture=args.bg_texture,
        math_rule=args.math,
        product_name=args.name,
        typo_style=args.typo,
        color_palette=args.colors,
        tags=args.tags,
        output_dir=args.output_dir,
        args=args,
        gem_api_key=gem_api_key,
        height=args.height,
        width=args.width,
    )


# ---------- MAIN ENTRY ----------
if __name__ == "__main__":
    # build pyinstaller --onefile --console gem.py
    main()