from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# Permet de mieux gérer certaines images légèrement tronquées
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Dataset images

class AirbnbImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        listing_id_clean = image_path.stem

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)

            return image, listing_id_clean, str(image_path), None

        except Exception as e:
            return None, listing_id_clean, str(image_path), str(e)


def collate_fn(batch):
    images = []
    ids = []
    paths = []
    failed = []

    for image, listing_id_clean, image_path, error in batch:
        if image is None:
            failed.append({
                "listing_id_clean": listing_id_clean,
                "image_path": image_path,
                "error": error
            })
        else:
            images.append(image)
            ids.append(listing_id_clean)
            paths.append(image_path)

    if len(images) == 0:
        return None, [], [], failed

    images = torch.stack(images)
    return images, ids, paths, failed

# Outils

def save_text_report(report_path, report_lines):
    with open(report_path, "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(str(line) + "\n")


def load_expected_ids(split_path):
    split_path = Path(split_path)

    if not split_path.exists():
        return None, None

    split_df = pd.read_csv(split_path, dtype=str)

    if "listing_id_clean" in split_df.columns:
        id_col = "listing_id_clean"
    elif "id_clean" in split_df.columns:
        id_col = "id_clean"
    else:
        raise ValueError(
            "Le fichier split existe, mais aucune colonne ID n'a été trouvée. "
            "Colonnes attendues : listing_id_clean ou id_clean."
        )

    split_df[id_col] = split_df[id_col].astype(str).str.strip()
    expected_ids = set(split_df[id_col])

    return split_df, expected_ids

# Fonction principale

def main(args):
    start_time = time.time()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if not image_dir.exists():
        raise FileNotFoundError(f"Dossier images introuvable : {image_dir}")

    image_paths = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]

    # ordre fixe et reproductible.
    # L'ordre du NPY suivra exactement cet ordre, en retirant seulement les images non lisibles.
    image_paths = sorted(image_paths, key=lambda p: p.stem)

    if len(image_paths) == 0:
        raise RuntimeError(f"Aucune image trouvée dans le dossier : {image_dir}")

    print("\n========== EXTRACTION EFFICIENTNET-B0 ==========\n")
    print(f"Dossier images : {image_dir}")
    print(f"Nombre d'images trouvées : {len(image_paths)}")
    print(f"Dossier de sortie : {output_dir}")

    # Vérification avec split_listing_ids.csv

    split_df, expected_ids = load_expected_ids(args.split_path)

    if expected_ids is not None:
        image_ids = set(p.stem for p in image_paths)
        missing_image_files = sorted(expected_ids - image_ids)
        extra_image_files = sorted(image_ids - expected_ids)

        print("\n========== CONTROLE AVEC SPLIT ==========\n")
        print(f"Fichier split : {args.split_path}")
        print(f"Nombre d'annonces attendues dans le split : {len(expected_ids)}")
        print(f"Images présentes dans le dossier : {len(image_ids)}")
        print(f"Images absentes du dossier : {len(missing_image_files)}")
        print(f"Images en trop dans le dossier : {len(extra_image_files)}")

        if len(missing_image_files) > 0:
            pd.DataFrame({"listing_id_clean": missing_image_files}).to_csv(
                output_dir / "images_absentes_du_dossier.csv",
                index=False,
                encoding="utf-8-sig"
            )

        if len(extra_image_files) > 0:
            pd.DataFrame({"listing_id_clean": extra_image_files}).to_csv(
                output_dir / "images_en_trop_dans_dossier.csv",
                index=False,
                encoding="utf-8-sig"
            )
    # Device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n========== DEVICE ==========\n")
    print(f"Device utilisé : {device}")

    if torch.cuda.is_available():
        print(f"GPU détecté : {torch.cuda.get_device_name(0)}")
    else:
        print("Attention : aucun GPU CUDA détecté. L'extraction sera beaucoup plus lente.")


    # Modèle EfficientNet-B0

    print("\nChargement de EfficientNet-B0 pré-entraîné...")

    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # On supprime la couche finale de classification.
    # La sortie devient un embedding visuel de dimension 1280.
    model.classifier = nn.Identity()

    model = model.to(device)
    model.eval()

    transform = weights.transforms()

    dataset = AirbnbImageDataset(
        image_paths=image_paths,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    all_embeddings = []
    all_ids = []
    all_paths = []
    failed_images = []

    print("\nExtraction des embeddings...\n")

    with torch.no_grad():
        for images, ids, paths, failed in tqdm(dataloader):
            failed_images.extend(failed)

            if images is None:
                continue

            images = images.to(device, non_blocking=True)

            embeddings = model(images)
            embeddings = embeddings.cpu().numpy().astype(np.float32)

            all_embeddings.append(embeddings)
            all_ids.extend(ids)
            all_paths.extend(paths)

    if len(all_embeddings) == 0:
        raise RuntimeError("Aucun embedding n'a été extrait.")

    embeddings_array = np.vstack(all_embeddings)
    # Contrôles internes

    if embeddings_array.shape[0] != len(all_ids):
        raise RuntimeError(
            "Problème d'alignement : le nombre de lignes du NPY ne correspond pas au nombre d'IDs."
        )

    if embeddings_array.shape[0] != len(all_paths):
        raise RuntimeError(
            "Problème d'alignement : le nombre de lignes du NPY ne correspond pas au nombre de chemins images."
        )

    print("\n========== RESULTAT EXTRACTION ==========\n")
    print(f"Nombre d'embeddings extraits : {embeddings_array.shape[0]}")
    print(f"Dimension de chaque embedding : {embeddings_array.shape[1]}")
    print(f"Images non lisibles : {len(failed_images)}")

    if embeddings_array.shape[1] != 1280:
        print("Attention : la dimension attendue avec EfficientNet-B0 est normalement 1280.")

    # 1. Sauvegarde NPY

    npy_path = output_dir / "efficientnet_b0_embeddings.npy"
    np.save(npy_path, embeddings_array)

    print(f"\nFichier NPY sauvegardé : {npy_path}")
    # 2. Sauvegarde fichier IDs aligné avec le NPY

    df_ids = pd.DataFrame({
        "row_npy": np.arange(len(all_ids), dtype=int),
        "listing_id_clean": all_ids,
        "image_path": all_paths
    })

    ids_path = output_dir / "efficientnet_b0_embeddings_ids.csv"
    df_ids.to_csv(ids_path, index=False, encoding="utf-8-sig")

    print(f"Fichier IDs aligné sauvegardé : {ids_path}")
    # 3. Sauvegarde CSV complet avec embeddings

    if args.save_full_csv:
        print("\nCréation du fichier CSV complet avec les 1280 embeddings...")

        emb_columns = [f"img_emb_{i}" for i in range(embeddings_array.shape[1])]

        df_embeddings = pd.DataFrame(
            embeddings_array,
            columns=emb_columns
        )

        df_embeddings.insert(0, "image_path", all_paths)
        df_embeddings.insert(0, "listing_id_clean", all_ids)
        df_embeddings.insert(0, "row_npy", np.arange(len(all_ids), dtype=int))

        csv_path = output_dir / "efficientnet_b0_embeddings.csv"
        df_embeddings.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
            float_format="%.7g"
        )

        print(f"Fichier CSV complet sauvegardé : {csv_path}")
    else:
        csv_path = None
        print("\nCSV complet non sauvegardé car save_full_csv=False.")

    # 4. Sauvegarde images non lisibles

    failed_path = output_dir / "images_non_lisibles.csv"

    if len(failed_images) > 0:
        pd.DataFrame(failed_images).to_csv(
            failed_path,
            index=False,
            encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(columns=["listing_id_clean", "image_path", "error"]).to_csv(
            failed_path,
            index=False,
            encoding="utf-8-sig"
        )

    print(f"Fichier images non lisibles sauvegardé : {failed_path}")

    # 5. Comparaison finale avec split

    missing_embedding_ids = []
    extra_embedding_ids = []

    if expected_ids is not None:
        embedding_ids = set(df_ids["listing_id_clean"].astype(str))

        missing_embedding_ids = sorted(expected_ids - embedding_ids)
        extra_embedding_ids = sorted(embedding_ids - expected_ids)

        pd.DataFrame({"listing_id_clean": missing_embedding_ids}).to_csv(
            output_dir / "annonces_sans_embedding.csv",
            index=False,
            encoding="utf-8-sig"
        )

        pd.DataFrame({"listing_id_clean": extra_embedding_ids}).to_csv(
            output_dir / "annonces_embeddings_en_trop.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print("\n========== COMPARAISON FINALE AVEC SPLIT ==========\n")
        print(f"Annonces attendues : {len(expected_ids)}")
        print(f"Annonces avec embedding : {len(embedding_ids)}")
        print(f"Annonces sans embedding : {len(missing_embedding_ids)}")
        print(f"Embeddings en trop : {len(extra_embedding_ids)}")

    # 6. Rapport final

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60

    report_lines = [
        "Rapport extraction embeddings EfficientNet-B0",
        "=" * 80,
        "",
        f"Dossier images : {image_dir}",
        f"Dossier sortie : {output_dir}",
        f"Nombre d'images trouvées au départ : {len(image_paths)}",
        f"Nombre d'embeddings extraits : {embeddings_array.shape[0]}",
        f"Dimension embeddings : {embeddings_array.shape[1]}",
        f"Images non lisibles : {len(failed_images)}",
        "",
        "Fichiers générés",
        "-" * 80,
        f"NPY embeddings : {npy_path}",
        f"IDs alignés NPY : {ids_path}",
        f"CSV complet : {csv_path if csv_path is not None else 'non sauvegardé'}",
        f"Images non lisibles : {failed_path}",
        "",
        "Contrôle split",
        "-" * 80,
    ]

    if expected_ids is not None:
        report_lines.extend([
            f"Nombre d'annonces attendues : {len(expected_ids)}",
            f"Nombre d'annonces avec embedding : {len(set(df_ids['listing_id_clean'].astype(str)))}",
            f"Nombre d'annonces sans embedding : {len(missing_embedding_ids)}",
            f"Nombre d'embeddings en trop : {len(extra_embedding_ids)}",
        ])
    else:
        report_lines.append("Aucun split fourni ou split introuvable.")

    report_lines.extend([
        "",
        "Temps d'exécution",
        "-" * 80,
        f"Temps total secondes : {elapsed:.2f}",
        f"Temps total minutes : {elapsed_min:.2f}",
        "",
        "Remarque",
        "-" * 80,
        "Le fichier efficientnet_b0_embeddings.npy contient uniquement les vecteurs numériques.",
        "Le fichier efficientnet_b0_embeddings_ids.csv relie chaque ligne du NPY à listing_id_clean.",
        "Pour CatBoost, il est conseillé d'utiliser le NPY + le fichier IDs plutôt que le gros CSV complet.",
    ])

    report_path = output_dir / "rapport_extraction_efficientnet_b0.txt"
    save_text_report(report_path, report_lines)

    config_path = output_dir / "config_extraction_efficientnet_b0.json"
    config = {
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "split_path": str(args.split_path),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "save_full_csv": args.save_full_csv,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_images_found": len(image_paths),
        "n_embeddings": int(embeddings_array.shape[0]),
        "embedding_dim": int(embeddings_array.shape[1]),
        "n_failed_images": len(failed_images),
        "n_missing_embeddings_vs_split": len(missing_embedding_ids) if expected_ids is not None else None,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n========== EXTRACTION TERMINEE ==========\n")
    print(f"Rapport sauvegardé : {report_path}")
    print(f"Configuration sauvegardée : {config_path}")
    print(f"Temps total : {elapsed_min:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        default="London",
        help="Dossier contenant les images."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="Resultat2",
        help="Dossier où sauvegarder les embeddings."
    )

    parser.add_argument(
        "--split_path",
        type=str,
        default="split_listing_ids.csv",
        help="Fichier split contenant les IDs attendus."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Nombre d'images traitées par batch."
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Nombre de workers pour charger les images."
    )

    parser.add_argument(
        "--save_full_csv",
        action="store_true",
        help="Sauvegarder aussi le gros CSV complet avec les 1280 embeddings."
    )

    args = parser.parse_args()
    main(args)