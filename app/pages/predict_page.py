"""Page 2 — Try the Model: load a trained model and make predictions."""
import os
import sys
import glob
from datetime import datetime
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.device import get_device
from src.models import build_model


# ── MNIST class labels ───────────────────────────────────────────
MNIST_CLASSES = [str(i) for i in range(10)]

# ── CIFAR-10 class labels ───────────────────────────────────────
CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ── California Housing feature descriptions ──────────────────────
HOUSING_FEATURES = {
    "MedInc":     {"label": "Median Income (x$10k)",      "min": 0.5,   "max": 15.0,   "default": 3.87,    "step": 0.1,
                   "hint": "Typical: $30k-$50k (enter 3.0-5.0). High-income areas: 8.0+"},
    "HouseAge":   {"label": "House Age (years)",          "min": 1.0,   "max": 52.0,   "default": 29.0,    "step": 1.0,
                   "hint": "1 = brand new, 52 = oldest in dataset. Median is ~29 years."},
    "AveRooms":   {"label": "Avg Rooms per Household",    "min": 1.0,   "max": 15.0,   "default": 5.4,     "step": 0.1,
                   "hint": "Typical: 4-7 rooms. Studios ~2, large homes 8+."},
    "AveBedrms":  {"label": "Avg Bedrooms per Household", "min": 0.5,   "max": 5.0,    "default": 1.1,     "step": 0.1,
                   "hint": "Typical: 1.0-1.2. This is the average across the census block."},
    "Population": {"label": "Block Population",           "min": 3.0,   "max": 5000.0, "default": 1425.0,  "step": 50.0,
                   "hint": "People in this census block. Typical: 500-2500."},
    "AveOccup":   {"label": "Avg Occupants per Household","min": 1.0,   "max": 10.0,   "default": 3.1,     "step": 0.1,
                   "hint": "Typical: 2-4 people per household."},
    "Latitude":   {"label": "Latitude",                   "min": 32.5,  "max": 42.0,   "default": 35.63,   "step": 0.01,
                   "hint": "SF: 37.77, LA: 34.05, San Diego: 32.72, Sacramento: 38.58"},
    "Longitude":  {"label": "Longitude",                  "min": -124.5,"max": -114.0,  "default": -119.57, "step": 0.01,
                   "hint": "SF: -122.42, LA: -118.24, San Diego: -117.16, Sacramento: -121.49"},
}

HOUSING_PRESETS = {
    "Custom (enter your own)": None,
    "Suburban Home (LA area)": {
        "MedInc": 5.0, "HouseAge": 25, "AveRooms": 6.5, "AveBedrms": 1.1,
        "Population": 1200, "AveOccup": 2.8, "Latitude": 34.05, "Longitude": -118.25,
    },
    "Urban Apartment (SF)": {
        "MedInc": 3.5, "HouseAge": 35, "AveRooms": 4.0, "AveBedrms": 1.0,
        "Population": 3000, "AveOccup": 3.5, "Latitude": 37.77, "Longitude": -122.42,
    },
    "Luxury Coastal (Malibu)": {
        "MedInc": 10.0, "HouseAge": 15, "AveRooms": 8.0, "AveBedrms": 1.5,
        "Population": 800, "AveOccup": 2.5, "Latitude": 33.95, "Longitude": -118.45,
    },
    "Rural Inland (Central Valley)": {
        "MedInc": 2.5, "HouseAge": 30, "AveRooms": 5.0, "AveBedrms": 1.1,
        "Population": 600, "AveOccup": 3.2, "Latitude": 36.75, "Longitude": -119.77,
    },
}

# Scaler stats (mean, std) for California Housing features.
# These are the population-level stats — close enough for the UI.
# The real scaler is fit on the training split, but these are very close.
HOUSING_MEANS = np.array([3.8707, 28.6395, 5.4290, 1.0968, 1425.4767, 3.0707, 35.6319, -119.5697])
HOUSING_STDS  = np.array([1.8998, 12.5856, 2.4742, 0.4739, 1132.4622, 10.3860, 2.1360, 2.0035])


def _find_trained_models() -> dict:
    """Scan runs/*/models/*.pt and return enriched model info.

    Returns: {display_name: {"path", "metric", "task", "optimizer", "lr", "seed", "epochs", "experiment", "trained_at"}}
    """
    all_optimizers = ["sgd_momentum", "nesterov", "adagrad", "rmsprop", "adam", "adamw", "sgd"]
    raw_models = []
    pattern = os.path.join("runs", "*", "models", "*.pt")

    for path in sorted(glob.glob(pattern)):
        parts = path.replace("\\", "/").split("/")
        experiment = parts[1]
        filename = parts[-1].replace(".pt", "")

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue

        summary = ckpt.get("summary", {})
        meta = ckpt.get("meta", {})
        task = meta.get("task", "classification")
        metric = summary.get("best_test_metric", 0.0)
        opt = summary.get("optimizer", "")
        lr = summary.get("lr", 0.0)
        seed = summary.get("seed", 0)
        ep = summary.get("epochs", ckpt.get("epochs", 0))

        trained_at = _parse_timestamp_from_filename(filename)

        if not opt:
            for o in all_optimizers:
                if filename.startswith(o + "_lr"):
                    opt = o
                    break

        raw_models.append({
            "path": path,
            "experiment": experiment,
            "optimizer": opt,
            "lr": lr,
            "seed": seed,
            "epochs": ep,
            "metric": metric,
            "task": task,
            "trained_at": trained_at,
        })

    best_by_experiment = {}
    for m in raw_models:
        exp = m["experiment"]
        if exp not in best_by_experiment or m["metric"] > best_by_experiment[exp]:
            best_by_experiment[exp] = m["metric"]

    models = {}
    for m in raw_models:
        metric_label = "Acc" if m["task"] == "classification" else "R²"
        is_best = (m["metric"] == best_by_experiment.get(m["experiment"], None))
        prefix = "🏆 " if is_best else ""
        ep_str = f", {m['epochs']}ep" if m["epochs"] else ""
        ts_str = f" [{m['trained_at']}]" if m["trained_at"] != "No timestamp" else ""
        display = f"{prefix}{m['optimizer']} | lr={m['lr']} | seed={m['seed']}{ep_str} | {metric_label}: {m['metric']:.4f}{ts_str}"
        display_key = f"{m['experiment']} / {display}"
        models[display_key] = m

    return models


def _load_model(path: str, device):
    """Load a saved model checkpoint and return (model, checkpoint)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_name"], checkpoint["meta"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _predict_mnist(model, device):
    """MNIST prediction UI — draw a digit or upload an image."""
    st.markdown("### ✏️ Draw a digit or upload an image")

    input_method = st.radio(
        "Input method", ["Draw on canvas", "Upload image"],
        horizontal=True, key="mnist_input",
    )

    img_array = None

    if input_method == "Draw on canvas":
        try:
            from streamlit_drawable_canvas import st_canvas

            st.markdown("Draw a digit (0–9) in the box below:")
            canvas = st_canvas(
                fill_color="black",
                stroke_width=18,
                stroke_color="white",
                background_color="black",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="mnist_canvas",
            )

            if canvas.image_data is not None:
                # Canvas returns RGBA, convert to grayscale
                img = Image.fromarray(canvas.image_data.astype("uint8"), "RGBA")
                img = img.convert("L")  # grayscale
                # Check if anything was drawn (not all black)
                if np.array(img).max() > 10:
                    img_array = np.array(img)
                else:
                    st.info("Draw something above, then predictions will appear below.")
                    return

        except ImportError:
            st.warning(
                "Canvas not available. Install it with: "
                "`pip install streamlit-drawable-canvas`\n\n"
                "Using upload mode instead."
            )
            input_method = "Upload image"

    if input_method == "Upload image":
        uploaded = st.file_uploader(
            "Upload a digit image (any size, will be resized to 28×28)",
            type=["png", "jpg", "jpeg", "bmp"],
            key="mnist_upload",
        )
        if uploaded is None:
            st.info("Upload an image of a handwritten digit.")
            return

        img = Image.open(uploaded).convert("L")
        img_array = np.array(img)
        st.image(img, caption="Uploaded image", width=200)

    if img_array is None:
        return

    # ── Preprocess to match MNIST format ─────────────────────────
    # Resize to 28x28
    img_pil = Image.fromarray(img_array)
    img_pil = img_pil.resize((28, 28), Image.LANCZOS)
    img_np = np.array(img_pil, dtype=np.float32)

    # Normalize same as training: (pixel - mean) / std
    img_np = img_np / 255.0
    img_np = (img_np - 0.1307) / 0.3081

    # Convert to tensor: (1, 1, 28, 28) — batch=1, channels=1
    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

    # ── Predict ──────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]  # convert to probabilities

    predicted_class = probs.argmax().item()
    confidence = probs[predicted_class].item()

    # ── Display results ──────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        # Show the preprocessed image
        st.image(
            img_pil.resize((140, 140), Image.NEAREST),
            caption="Preprocessed (28×28)",
        )
        st.metric("Prediction", f"{predicted_class}", f"{confidence:.1%} confident")

    with col2:
        # Confidence bar chart
        st.markdown("**Confidence per class:**")
        chart_data = {
            "Digit": MNIST_CLASSES,
            "Confidence": [p.item() for p in probs],
        }
        import pandas as pd
        df = pd.DataFrame(chart_data)
        st.bar_chart(df.set_index("Digit"), use_container_width=True)


def _predict_cifar(model, device):
    """CIFAR-10 prediction UI — upload a color image."""
    st.markdown("### 🖼️ Upload an image to classify")
    st.caption("CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")

    uploaded = st.file_uploader(
        "Upload an image (any size, will be resized to 32×32)",
        type=["png", "jpg", "jpeg", "bmp"],
        key="cifar_upload",
    )

    if uploaded is None:
        st.info("Upload an image to classify.")
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", width=200)

    # ── Preprocess to match CIFAR-10 format ──────────────────────
    img_resized = img.resize((32, 32), Image.LANCZOS)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    # Normalize per channel (same as training)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    img_np = (img_np - mean) / std

    # (H, W, C) -> (C, H, W) -> (1, C, H, W)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # ── Predict ──────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    predicted_idx = probs.argmax().item()
    predicted_class = CIFAR_CLASSES[predicted_idx]
    confidence = probs[predicted_idx].item()

    # ── Display results ──────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(
            img_resized.resize((140, 140), Image.NEAREST),
            caption="Preprocessed (32×32)",
        )
        st.metric("Prediction", predicted_class, f"{confidence:.1%} confident")

    with col2:
        st.markdown("**Confidence per class:**")
        import pandas as pd
        chart_data = {
            "Class": CIFAR_CLASSES,
            "Confidence": [p.item() for p in probs],
        }
        df = pd.DataFrame(chart_data)
        st.bar_chart(df.set_index("Class"), use_container_width=True)


def _predict_housing(model, device):
    """California Housing prediction UI — empty inputs, map location picker."""
    import pandas as pd
    import folium
    from streamlit_folium import st_folium

    st.markdown("### 🏠 Predict median house value")
    st.caption("Select a preset, pick a location, or enter values manually.")

    # ── Callbacks (run BEFORE the script reruns) ─────────────────
    def _apply_location(lat, lon):
        st.session_state["housing_Latitude"] = lat
        st.session_state["housing_Longitude"] = lon
        st.session_state.pop("_clicked_lat", None)
        st.session_state.pop("_clicked_lng", None)

    def _apply_preset():
        chosen = st.session_state.get("housing_preset")
        preset_vals = HOUSING_PRESETS.get(chosen)
        for feat_name in HOUSING_FEATURES:
            key = f"housing_{feat_name}"
            if preset_vals:
                st.session_state[key] = float(preset_vals[feat_name])
            else:
                st.session_state.pop(key, None)

    # ── Preset selector ──────────────────────────────────────────
    preset_name = st.selectbox(
        "Quick presets",
        list(HOUSING_PRESETS.keys()),
        key="housing_preset",
        on_change=_apply_preset,
    )
    preset = HOUSING_PRESETS[preset_name]

    # ── Location quick-select buttons ────────────────────────────
    LOCATIONS = {
        "San Francisco": (37.77, -122.42),
        "Los Angeles": (34.05, -118.24),
        "San Diego": (32.72, -117.16),
        "Sacramento": (38.58, -121.49),
        "Fresno": (36.75, -119.77),
    }
    loc_cols = st.columns(len(LOCATIONS))
    for idx, (city, (clat, clon)) in enumerate(LOCATIONS.items()):
        with loc_cols[idx]:
            st.button(city, key=f"loc_{city}",
                      on_click=_apply_location, args=(clat, clon),
                      use_container_width=True)

    # ── Interactive map ──────────────────────────────────────────
    _CA_CENTER = (36.5, -119.5)
    cur_lat = st.session_state.get("housing_Latitude")
    cur_lon = st.session_state.get("housing_Longitude")
    map_lat = cur_lat if cur_lat is not None else _CA_CENTER[0]
    map_lon = cur_lon if cur_lon is not None else _CA_CENTER[1]

    fmap = folium.Map(location=[map_lat, map_lon], zoom_start=6)
    if cur_lat is not None and cur_lon is not None:
        folium.Marker(
            [map_lat, map_lon],
            popup=f"Lat: {map_lat:.2f}, Lon: {map_lon:.2f}",
            icon=folium.Icon(color="red", icon="home", prefix="fa"),
        ).add_to(fmap)
    st.caption("Click on the map to select a location:")
    map_data = st_folium(fmap, height=350, key="housing_map")

    if map_data and map_data.get("last_clicked"):
        clicked_lat = round(map_data["last_clicked"]["lat"], 2)
        clicked_lng = round(map_data["last_clicked"]["lng"], 2)
        st.session_state["_clicked_lat"] = max(32.5, min(42.0, clicked_lat))
        st.session_state["_clicked_lng"] = max(-124.5, min(-114.0, clicked_lng))

    if "_clicked_lat" in st.session_state:
        d_lat = st.session_state["_clicked_lat"]
        d_lng = st.session_state["_clicked_lng"]
        mc1, mc2 = st.columns([3, 1])
        with mc1:
            st.info(f"Clicked: **Lat {d_lat:.2f}, Lon {d_lng:.2f}**")
        with mc2:
            st.button("Use this location", key="apply_map_loc",
                      on_click=_apply_location, args=(d_lat, d_lng),
                      type="primary")

    # ── Feature inputs (empty by default, presets fill them) ─────
    st.markdown("---")
    values = []
    cols = st.columns(2)

    for i, (feat_name, feat_info) in enumerate(HOUSING_FEATURES.items()):
        with cols[i % 2]:
            if preset and f"housing_{feat_name}" not in st.session_state:
                init_val = float(preset[feat_name])
            else:
                init_val = None

            val = st.number_input(
                feat_info["label"],
                min_value=float(feat_info["min"]),
                max_value=float(feat_info["max"]),
                value=init_val,
                step=float(feat_info["step"]),
                key=f"housing_{feat_name}",
                placeholder=feat_info["hint"],
            )
            values.append(val)

    # ── Check all values provided ────────────────────────────────
    missing = [list(HOUSING_FEATURES.values())[i]["label"]
               for i, v in enumerate(values) if v is None]
    if missing:
        st.info(f"Enter values for: {', '.join(missing)}")
        return

    # ── Standardize and predict ──────────────────────────────────
    raw = np.array(values, dtype=np.float32)
    standardized = (raw - HOUSING_MEANS) / HOUSING_STDS
    tensor = torch.from_numpy(standardized).unsqueeze(0).float().to(device)

    with torch.no_grad():
        prediction = model(tensor)

    predicted_value = prediction.item()
    price_dollars = predicted_value * 100_000

    # ── Display results ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("Predicted Median House Value")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Price", f"${price_dollars:,.0f}")
    with col2:
        if price_dollars < 100_000:
            st.metric("Range", "Below Average", help="Likely rural or inland area")
        elif price_dollars < 250_000:
            st.metric("Range", "Near Median", help="Typical California census block")
        elif price_dollars < 500_000:
            st.metric("Range", "Above Average", help="Desirable neighborhood")
        else:
            st.metric("Range", "Premium", help="Coastal or high-income area")

    st.caption(f"Raw model output: {predicted_value:.4f} (units of $100k)")

    with st.expander("View input summary"):
        feature_df = pd.DataFrame({
            "Feature": [f["label"] for f in HOUSING_FEATURES.values()],
            "Your Value": [f"{v:.2f}" for v in values],
            "Dataset Average": [f"{avg:.2f}" for avg in HOUSING_MEANS],
        })
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

def _evaluate_quick(model, test_loader, criterion, device, task):
    """Quick evaluation for the retrain feature."""
    from sklearn.metrics import r2_score

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            if task == "regression":
                by = by.float()
                if by.dim() == 1:
                    by = by.unsqueeze(1)
            out = model(bx)
            loss = criterion(out, by)
            running_loss += loss.item() * bx.size(0)
            total += bx.size(0)
            if task == "classification":
                correct += (out.argmax(1) == by).sum().item()
            else:
                all_preds.extend(out.cpu().numpy().flatten())
                all_targets.extend(by.cpu().numpy().flatten())

    avg_loss = running_loss / total
    if task == "classification":
        metric = correct / total
    else:
        metric = r2_score(all_targets, all_preds)
    return avg_loss, metric

import re

EXPERIMENT_LABELS = {
    "mnist_cnn": "MNIST — SmallCNN",
    "cifar_resnet8": "CIFAR-10 — ResNet8",
    "housing_mlp": "Housing — DeepMLP",
}

_TS_PATTERN = re.compile(r"_(\d{8}_\d{6})$")


def _parse_timestamp_from_filename(filename):
    """Extract timestamp from filename like 'adam_lr0.001_seed42_20260513_143022'."""
    match = _TS_PATTERN.search(filename.replace(".pt", ""))
    if match:
        raw = match.group(1)
        try:
            dt = datetime.strptime(raw, "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return "No timestamp"


def _section_delete_models():
    """Expandable section to delete trained models."""
    with st.expander("Manage Trained Models", expanded=False):
        all_models = _find_trained_models()

        if not all_models:
            st.info("No trained models found.")
            return

        # Group by experiment
        by_experiment = {}
        for display_key, info in all_models.items():
            exp = info["experiment"]
            if exp not in by_experiment:
                by_experiment[exp] = []
            ts = _parse_timestamp_from_filename(os.path.basename(info["path"]))
            metric_label = "Acc" if info["task"] == "classification" else "R²"
            by_experiment[exp].append({
                "display": f"{info['optimizer']} (lr={info['lr']}, seed={info['seed']}) — {metric_label}: {info['metric']:.4f}",
                "timestamp": ts,
                "path": info["path"],
                "key": display_key,
            })

        # Sort each group by timestamp (newest first)
        for exp in by_experiment:
            by_experiment[exp].sort(key=lambda x: x["timestamp"], reverse=True)

        selected_exp = st.selectbox(
            "Experiment",
            list(by_experiment.keys()),
            format_func=lambda x: EXPERIMENT_LABELS.get(x, x),
            key="delete_exp",
        )

        models_in_exp = by_experiment[selected_exp]

        delete_options = [
            f"[{m['timestamp']}]  {m['display']}"
            for m in models_in_exp
        ]

        selected_to_delete = st.multiselect(
            "Select models to delete",
            options=range(len(delete_options)),
            format_func=lambda i: delete_options[i],
            key="delete_models_select",
        )

        if selected_to_delete:
            st.warning(f"You are about to delete {len(selected_to_delete)} model(s). This cannot be undone.")
            if st.button("Delete Selected Models", type="primary", key="confirm_delete"):
                deleted = 0
                for idx in selected_to_delete:
                    path = models_in_exp[idx]["path"]
                    try:
                        os.remove(path)
                        deleted += 1
                    except OSError as e:
                        st.error(f"Failed to delete {os.path.basename(path)}: {e}")
                if deleted:
                    st.success(f"Deleted {deleted} model(s).")
                    st.rerun()


def _section_compare_models(ds_info, available, device):
    """Side-by-side comparison of two models on the same input."""
    with st.expander("Compare Two Models Side-by-Side", expanded=False):
        experiment_models = {k: v for k, v in available.items()
                            if v["experiment"] == ds_info["experiment"]}

        if len(experiment_models) < 2:
            st.info("Train at least 2 models for this dataset to enable comparison.")
            return

        sorted_models = dict(sorted(experiment_models.items(),
                                     key=lambda x: x[1]["metric"], reverse=True))
        display_names = list(sorted_models.keys())
        short_names = [k.split(" / ", 1)[1] for k in display_names]

        col_a, col_b = st.columns(2)
        with col_a:
            idx_a = st.selectbox("Model A", range(len(short_names)),
                                 format_func=lambda i: short_names[i],
                                 key="cmp_model_a")
        with col_b:
            default_b = min(1, len(short_names) - 1)
            idx_b = st.selectbox("Model B", range(len(short_names)),
                                 format_func=lambda i: short_names[i],
                                 index=default_b, key="cmp_model_b")

        info_a = sorted_models[display_names[idx_a]]
        info_b = sorted_models[display_names[idx_b]]

        model_a, ckpt_a = _load_model(info_a["path"], device)
        model_b, ckpt_b = _load_model(info_b["path"], device)

        dataset = ds_info["dataset"]

        if dataset in ("mnist", "cifar10"):
            uploaded = st.file_uploader(
                "Upload an image for both models to classify",
                type=["png", "jpg", "jpeg", "bmp"],
                key="cmp_upload",
            )
            if uploaded is None:
                st.info("Upload an image to compare predictions.")
                return

            img = Image.open(uploaded)

            if dataset == "mnist":
                img_gray = img.convert("L")
                img_resized = img_gray.resize((28, 28), Image.LANCZOS)
                img_np = np.array(img_resized, dtype=np.float32) / 255.0
                img_np = (img_np - 0.1307) / 0.3081
                tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
                classes = MNIST_CLASSES
            else:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize((32, 32), Image.LANCZOS)
                img_np = np.array(img_resized, dtype=np.float32) / 255.0
                mean = np.array([0.4914, 0.4822, 0.4465])
                std = np.array([0.2470, 0.2435, 0.2616])
                img_np = (img_np - mean) / std
                tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
                classes = CIFAR_CLASSES

            with torch.no_grad():
                probs_a = F.softmax(model_a(tensor), dim=1)[0]
                probs_b = F.softmax(model_b(tensor), dim=1)[0]

            pred_a = probs_a.argmax().item()
            pred_b = probs_b.argmax().item()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Model A:** {info_a['optimizer']} (lr={info_a['lr']})")
                st.metric("Prediction", classes[pred_a], f"{probs_a[pred_a].item():.1%}")
                import pandas as pd_cmp
                df_a = pd_cmp.DataFrame({"Class": classes, "Confidence": [p.item() for p in probs_a]})
                st.bar_chart(df_a.set_index("Class"), use_container_width=True)

            with col2:
                st.markdown(f"**Model B:** {info_b['optimizer']} (lr={info_b['lr']})")
                st.metric("Prediction", classes[pred_b], f"{probs_b[pred_b].item():.1%}")
                df_b = pd_cmp.DataFrame({"Class": classes, "Confidence": [p.item() for p in probs_b]})
                st.bar_chart(df_b.set_index("Class"), use_container_width=True)

            if pred_a == pred_b:
                st.success(f"Both models agree: **{classes[pred_a]}**")
            else:
                st.warning(f"Models disagree: A says **{classes[pred_a]}**, B says **{classes[pred_b]}**")

        elif dataset == "california":
            st.caption("Both models will predict using the same housing features entered above.")
            values = []
            any_missing = False
            for feat_name, feat_info in HOUSING_FEATURES.items():
                key = f"housing_{feat_name}"
                val = st.session_state.get(key)
                if val is None:
                    any_missing = True
                    break
                values.append(float(val))
            if any_missing:
                st.info("Fill in all housing features above before comparing.")
                return

            raw = np.array(values, dtype=np.float32)
            standardized = (raw - HOUSING_MEANS) / HOUSING_STDS
            tensor = torch.from_numpy(standardized).unsqueeze(0).float().to(device)

            with torch.no_grad():
                pred_a = model_a(tensor).item() * 100_000
                pred_b = model_b(tensor).item() * 100_000

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Model A:** {info_a['optimizer']} (lr={info_a['lr']})")
                st.metric("Predicted Price", f"${pred_a:,.0f}")
            with col2:
                st.markdown(f"**Model B:** {info_b['optimizer']} (lr={info_b['lr']})")
                st.metric("Predicted Price", f"${pred_b:,.0f}")

            diff = abs(pred_a - pred_b)
            st.caption(f"Difference: ${diff:,.0f}")


def render():
    """Main render function for the Predict page."""
    st.title("🔮 Try the Model")
    st.markdown("Load a trained model and make predictions with new data.")

    # ── Dataset / optimizer selection ─────────────────────────────
    st.subheader("1. Select Dataset & Optimizer")

    dataset_options = {
        "MNIST (Digit Recognition)": {"dataset": "mnist", "model": "small_cnn", "experiment": "mnist_cnn"},
        "CIFAR-10 (Image Classification)": {"dataset": "cifar10", "model": "resnet8", "experiment": "cifar_resnet8"},
        "California Housing (Price Prediction)": {"dataset": "california", "model": "deep_mlp", "experiment": "housing_mlp"},
    }

    all_optimizers = ["sgd", "sgd_momentum", "nesterov", "adagrad", "rmsprop", "adam", "adamw"]

    selected_dataset = st.selectbox("Choose dataset", list(dataset_options.keys()))
    ds_info = dataset_options[selected_dataset]

    selected_optimizer = st.selectbox("Choose optimizer", all_optimizers, index=5)  # default adam

    # ── Find matching trained model ──────────────────────────────
    available = _find_trained_models()

    matching = {k: v for k, v in available.items()
                if v["experiment"] == ds_info["experiment"] and v["optimizer"] == selected_optimizer}

    # Sort by metric (best first)
    matching = dict(sorted(matching.items(), key=lambda x: x[1]["metric"], reverse=True))

    model = None
    checkpoint = None
    device = get_device()

    if matching:
        # Show dropdown with accuracy scores
        display_names = [k.split(" / ", 1)[1] for k in matching.keys()]
        full_keys = list(matching.keys())

        if len(display_names) > 1:
            selected_idx = st.selectbox(
                "Select a trained model:",
                range(len(display_names)),
                format_func=lambda i: display_names[i],
                key="model_picker",
            )
        else:
            selected_idx = 0
            st.success(f"Found trained model: {display_names[0]}")

        selected_info = matching[full_keys[selected_idx]]
        model_path = selected_info["path"]

        with st.spinner("Loading model..."):
            model, checkpoint = _load_model(model_path, device)

    else:
        st.warning(f"No trained **{selected_optimizer}** model found for **{selected_dataset}**.")

        # ── Retrain button ───────────────────────────────────────
        st.markdown("**Quick-train this model now?**")

        col1, col2, col3 = st.columns(3)
        with col1:
            retrain_lr = st.number_input("Learning rate", value=0.001, format="%.4f",
                                         min_value=0.00001, max_value=1.0, key="retrain_lr")
        with col2:
            retrain_epochs = st.number_input("Epochs", value=5, min_value=1, max_value=50,
                                             key="retrain_epochs")
        with col3:
            retrain_seed = st.number_input("Seed", value=42, min_value=0, max_value=9999,
                                           key="retrain_seed")

        if st.button(f"🚀 Train {selected_optimizer} on {ds_info['dataset']}", type="primary"):
            from src.seed import set_seed
            from src.datasets import build_dataset
            from src.models import build_model as build_model_fn
            from src.optim.registry import build_optimizer
            from src.bench.metrics import (
                Timer, MemoryProbe, optimizer_state_bytes,
                grad_norm as compute_grad_norm, param_norm as compute_param_norm,
                steps_to_threshold,
            )
            from sklearn.metrics import r2_score
            import torch.nn as nn

            set_seed(retrain_seed)

            with st.spinner("Loading dataset..."):
                batch_size = 128 if ds_info["dataset"] != "california" else 64
                train_loader, test_loader, meta = build_dataset(
                    ds_info["dataset"], {"batch_size": batch_size}
                )

            model = build_model_fn(ds_info["model"], meta).to(device)
            optimizer = build_optimizer(selected_optimizer, model.parameters(), {"lr": retrain_lr})

            task = meta["task"]
            criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

            progress = st.progress(0)
            status = st.empty()

            total_timer = Timer()
            memory_probe = MemoryProbe(device)
            test_metrics = []

            total_timer.start()

            for epoch in range(1, retrain_epochs + 1):
                model.train()
                memory_probe.reset_gpu_peak()
                running_loss = 0.0
                correct = 0
                total = 0

                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    if task == "regression":
                        by = by.float()
                        if by.dim() == 1:
                            by = by.unsqueeze(1)

                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * bx.size(0)
                    total += bx.size(0)
                    if task == "classification":
                        correct += (out.argmax(1) == by).sum().item()

                train_loss = running_loss / total
                train_metric = correct / total if task == "classification" else 0

                # Quick eval
                test_loss, test_metric = _evaluate_quick(model, test_loader, criterion, device, task)
                test_metrics.append(test_metric)

                progress.progress(epoch / retrain_epochs)
                mname = "Acc" if task == "classification" else "R²"
                status.text(f"Epoch {epoch}/{retrain_epochs} — Loss: {train_loss:.4f}, Test {mname}: {test_metric:.4f}")

            total_time = total_timer.stop()
            best_metric = max(test_metrics) if task == "classification" else min(test_metrics)

            summary = {
                "optimizer": selected_optimizer,
                "lr": retrain_lr,
                "seed": retrain_seed,
                "epochs": retrain_epochs,
                "total_time_s": round(total_time, 2),
                "best_test_metric": round(best_metric, 6),
                "final_test_metric": round(test_metrics[-1], 6),
                "steps_to_threshold": -1,
                "optimizer_state_bytes": optimizer_state_bytes(optimizer),
                "peak_gpu_mb": round(memory_probe.peak_gpu_mb(), 1),
                "peak_cpu_mb": round(memory_probe.cpu_rss_mb(), 0),
                "final_grad_norm": round(compute_grad_norm(model), 4),
                "final_param_norm": round(compute_param_norm(model), 4),
            }

            # Save model
            model_save_dir = os.path.join("runs", ds_info["experiment"], "models")
            os.makedirs(model_save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task = meta["task"]
            metric_tag = "acc" if task == "classification" else "r2"
            metric_val = best_metric
            model_fname = f"{selected_optimizer}_lr{retrain_lr}_seed{retrain_seed}_ep{retrain_epochs}_{metric_tag}{metric_val:.4f}_{timestamp}.pt"
            model_path = os.path.join(model_save_dir, model_fname)
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": ds_info["model"],
                "dataset": ds_info["dataset"],
                "optimizer": selected_optimizer,
                "lr": retrain_lr,
                "seed": retrain_seed,
                "epochs": retrain_epochs,
                "meta": meta,
                "summary": summary,
            }, model_path)

            checkpoint = {
                "dataset": ds_info["dataset"],
                "summary": summary,
                "meta": meta,
            }

            mname = "Accuracy" if task == "classification" else "R²"
            st.success(
                f"✅ Training complete! Best {mname}: {best_metric:.4f} | "
                f"Time: {total_time:.1f}s"
            )

    # ── Manage models ────────────────────────────────────────────
    st.markdown("---")
    _section_delete_models()

    # ── Show model info + prediction UI ──────────────────────────
    if model is None:
        return

    st.markdown("---")

    dataset = checkpoint.get("dataset", ds_info["dataset"])
    summary = checkpoint.get("summary", {})
    meta = checkpoint.get("meta", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset", dataset)
    with col2:
        st.metric("Optimizer", summary.get("optimizer", selected_optimizer))
    with col3:
        metric_name = "Accuracy" if meta.get("task") == "classification" else "R²"
        st.metric(f"Best {metric_name}", f"{summary.get('best_test_metric', 0):.4f}")
    with col4:
        st.metric("Training Time", f"{summary.get('total_time_s', 0):.1f}s")

    st.markdown("---")
    st.subheader("2. Make a Prediction")

    if dataset == "mnist":
        _predict_mnist(model, device)
    elif dataset == "cifar10":
        _predict_cifar(model, device)
    elif dataset == "california":
        _predict_housing(model, device)
    else:
        st.error(f"Unknown dataset: {dataset}")

    # ── Side-by-side comparison ──────────────────────────────────
    st.markdown("---")
    _section_compare_models(ds_info, available, device)

    # ── Export model ─────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Export Model to ONNX"):
        st.caption("Export for deployment with ONNX Runtime or other inference engines.")
        if st.button("Generate ONNX file", key="gen_onnx"):
            with st.spinner("Exporting..."):
                import io as _io
                try:
                    if dataset == "mnist":
                        dummy = torch.randn(1, 1, 28, 28).to(device)
                    elif dataset == "cifar10":
                        dummy = torch.randn(1, 3, 32, 32).to(device)
                    else:
                        dummy = torch.randn(1, 8).to(device)
                    buf = _io.BytesIO()
                    torch.onnx.export(
                        model, dummy, buf,
                        input_names=["input"], output_names=["output"],
                        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                    )
                    buf.seek(0)
                    opt_name = summary.get("optimizer", "model")
                    lr_val = summary.get("lr", 0)
                    st.session_state["_onnx_bytes"] = buf.getvalue()
                    st.session_state["_onnx_fname"] = f"{dataset}_{opt_name}_lr{lr_val}.onnx"
                except Exception as e:
                    st.error(f"Export failed: {e}")

        if "_onnx_bytes" in st.session_state:
            st.download_button(
                f"Download {st.session_state.get('_onnx_fname', 'model.onnx')}",
                st.session_state["_onnx_bytes"],
                file_name=st.session_state.get("_onnx_fname", "model.onnx"),
                mime="application/octet-stream",
                key="dl_onnx",
            )
            st.caption(f"Size: {len(st.session_state['_onnx_bytes']) / 1024:.0f} KB")