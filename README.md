# Projet de fin de module — Deep Learning (EMSI 2025–2026)

> **Conception, implémentation, comparaison et analyse critique de modèles de deep learning pour
> données tabulaires, images et séquences.**
> Module : Deep Learning · Filière Informatique · EMSI Casablanca · Travail individuel.

---

## 1. Idée générale du projet

L'efficacité d'un réseau de neurones ne dépend pas seulement de sa capacité, mais de l'**adéquation
entre son architecture et la structure statistique des données**. Ce projet le démontre, de bout en
bout et sous **PyTorch**, sur trois familles de données radicalement différentes :

| Données | Structure | Architecture étudiée | Dataset réel |
|---------|-----------|----------------------|--------------|
| **Tabulaire** | vecteur de variables, sans géométrie ni ordre | **MLP** | Wine Quality (vin rouge) |
| **Image** | grille 2D, localité + invariance par translation | **CNN** type LeNet | CIFAR-10 |
| **Séquence** | suite ordonnée de longueur variable, temporalité | **RNN / LSTM / GRU / Seq2Seq** | Tatoeba fra-eng |

Chaque partie comporte une **étude théorique**, une **implémentation**, une **étude expérimentale**,
une **analyse critique** et une **question de synthèse**. Une **synthèse transversale** relie les trois.
Le résultat central, vérifié expérimentalement (ex. CNN ≫ MLP sur l'image) : **chaque type de données
appelle un biais inductif particulier**.

Chaque notebook est **entièrement auto-contenu** : configuration, téléchargement des données, théorie,
code, expériences et analyse y figurent. Aucun chemin ni hyperparamètre n'est codé en dur — tout passe
par une dataclass `CONFIG` en tête de notebook.

---

## 2. Structure du dépôt

```
projet-fin-de-module-deep-learning/
│
├── README.md                              ← ce fichier (présentation + guide)
│
├── Partie1_MLP_WineQuality.ipynb          ← Partie I  : MLP / données tabulaires
├── Partie2_CNN_CIFAR10.ipynb              ← Partie II : CNN / images
├── Partie3_RNN_Seq2Seq_FraEng.ipynb       ← Partie III: RNN-LSTM-GRU-Seq2Seq / texte
├── Partie4_Synthese_Transversale.ipynb    ← Synthèse transversale finale (markdown)
│
├── Rapport_Projet_Deep_Learning.docx      ← Rapport scientifique (Word, version la plus complète)
├── Rapport_Projet_Deep_Learning.pdf       ← Rapport scientifique (PDF)
├── Rapport_Projet_Deep_Learning.md        ← Rapport scientifique (source Markdown)
├── abreviation.docx                       ← Liste des abréviations, sigles et symboles
│
├── figures/                               ← figures (PNG) utilisées par le rapport
│
├── _build/                                ← scripts générateurs des notebooks (outil, optionnel)
│   ├── gen_part1.py … gen_part4.py
│   ├── extract_figs.py                    ← extrait les figures des notebooks
│   └── convert_pdf.py                     ← convertit le rapport .md en .pdf
│
├── Projet_Deep-Learning_EMSI.pdf          ← sujet (fourni)
├── fiche_synthese_MLP_PyTorch.pdf         ← fiche de synthèse (fournie)
├── synthese_rnn_seq2seq.pdf               ← fiche de synthèse (fournie)
│
├── .gitignore                             ← exclut les caches régénérables (data/, models/…)
└── .gitattributes                         ← règles Git LFS pour les fichiers lourds
```

**Dossiers générés automatiquement à l'exécution** (donc *non versionnés* — voir `.gitignore`) :

```
data/      ← datasets téléchargés et mis en cache (Wine CSV, CIFAR-10, fra-eng)
models/    ← checkpoints sauvegardés (state_dict : mlp_wine_best.pt, cnn_cifar10.pt, seq2seq_fra_eng.pt)
```

---

## 3. Comment exécuter

> ⚙️ Interpréteur requis : **Python 3.13** (celui qui contient PyTorch).
> Dépendances : `torch`, `torchvision`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`,
> `jupyter`/`nbformat`. (Aucune dépendance à `nltk` — BLEU est réimplémenté — ni à l'API Kaggle.)

**Dans VSCode / Jupyter** : ouvrir un notebook, sélectionner le kernel Python 3.13, puis *Run All*.

**En ligne de commande** (exécute un notebook entier et enregistre les sorties) :

```bash
jupyter nbconvert --to notebook --execute --inplace Partie1_MLP_WineQuality.ipynb
```

Les parties sont **indépendantes** et peuvent être lancées dans n'importe quel ordre. Durées
indicatives sur **CPU** : Partie I ≈ 25 s · Partie II ≈ 4 min (télécharge CIFAR-10) · Partie III ≈ 5 min.

**Régénérer le rapport PDF** (après modification du `.md`) :

```bash
python _build/convert_pdf.py
```

---

## 4. Données et fichiers volumineux (Git LFS)

Les notebooks **téléchargent automatiquement** les jeux de données dans `data/` (mise en cache). Les
liens Kaggle du sujet exigeant une authentification, on utilise des **mirrors publics équivalents** :
UCI (Wine Quality), `torchvision` (CIFAR-10), mirror d2l (fra-eng) — sans changer la nature des données.

**Choix de versionnement.** Comme les données sont régénérables, `data/` et `models/` sont **exclus du
dépôt** (`.gitignore`) afin de garder le repo léger : inutile de versionner les ~170 Mo de CIFAR-10.
Le dépôt reste **complet et reproductible** : il suffit d'exécuter un notebook pour reconstituer
données et modèles.

**Git LFS** est néanmoins pré-configuré (`.gitattributes`) pour les binaires lourds (`*.pt`, `*.zip`,
`*.tar.gz`) : si vous choisissez de committer des modèles entraînés ou un jeu de données, ils seront
automatiquement gérés par LFS. Activation : `git lfs install` (une fois sur la machine).

---

## 5. Correspondance avec le cahier des charges

**Livrables (section 9 du sujet).**

| Exigé | Fourni |
|-------|--------|
| Rapport scientifique structuré | `Rapport_Projet_Deep_Learning.docx` / `.pdf` / `.md` (intro, objectifs, méthodologie, implémentation, résultats, interprétation, limites, conclusion + justification des choix) |
| Code source complet et commenté | les 4 notebooks `.ipynb` (+ générateurs `_build/`) |
| Notebook/script principal exécutable | les 4 notebooks `.ipynb` |
| Documentation / synthèse | `README.md`, `Partie4_Synthese_Transversale.ipynb`, `abreviation.docx` |
| Annexe expérimentale (courbes, tableaux, métriques, visualisations) | `figures/` + tableaux dans le rapport et les notebooks |

**Modèles exigés et fournis** : MLP (2 versions `Sequential` + classe), CNN type LeNet, RNN, LSTM, GRU,
Seq2Seq encodeur–décodeur — chacun répond à un point explicite du sujet. Éléments associés couverts :
3 initialisations + save/reload (Partie I) ; corr2d / max-pool / avg-pool « maison » + conv 1×1
(Partie II) ; gradient clipping, teacher forcing, perte masquée, décodage glouton **et** beam search,
BLEU + perplexité (Partie III).

---

## 6. Résultats principaux (CPU, sous-échantillon configurable)

- **Partie I** — accuracy test ≈ **0,73**, F1 ≈ 0,72 ; Xavier > gaussienne ≫ constante (symétrie non brisée).
- **Partie II** — **CNN 0,64 vs MLP 0,46** (à paramètres comparables) ; implémentations « maison » == PyTorch.
- **Partie III** — perplexité Seq2Seq ≈ **11,5** ; LSTM/GRU < RNN simple ; BLEU(beam) ≈ BLEU(glouton) ≈ 0,037.

> Les **tendances comparatives** (et non les valeurs absolues, bridées par le CPU) constituent le
> résultat scientifique. Pour reproduire à pleine échelle sur GPU, augmenter les champs de `CONFIG`
> (`train_subset`, `n_pairs`, `epochs`, …).

---

## 7. Environnement technique

Python 3.13 · PyTorch 2.11 (**CPU**) · torchvision 0.26 · pandas · scikit-learn · matplotlib · seaborn ·
Jupyter / nbformat. Reproductibilité assurée par graines fixées (`seed = 42`) et configurations
centralisées.
