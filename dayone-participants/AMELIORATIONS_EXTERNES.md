# Améliorations qui demandent une ressource externe (API, données, matériel)

Tout ce qui est dans ce dépôt tourne **hors ligne**, sur un portable (RTX 4060 8 Go) et sur la box de l'établissement.
Ce document liste ce qui pourrait encore faire monter la précision mais demande quelque chose que nous n'avions pas :
une clé d'API (Gemini…), des données à collecter, un GPU plus gros ou un compte Meta. Pour chaque piste : pourquoi
(l'écart mesuré qu'elle vise), comment (commandes prêtes), ce qui est permis côté confidentialité, et le gain attendu
(hypothèse tant que ce n'est pas mesuré).

## 0. Où est l'erreur aujourd'hui (mesuré)

| Constat | Mesure | Conséquence |
|---|---|---|
| Le recalage n'est **pas** le goulot | à sev 2, recalage oracle (homographie vraie) : champs remplis 0.705 vs 0.703 avec notre recalage | inutile d'investir dans la géométrie |
| La lecture des petites écritures floues l'est | au début de cette passe : champs remplis 0.94 (propre) → 0.70 (sev 2), 300 des 553 erreurs sev 2 dans le tableau des visites ; après (modèle v3, beam grammatical, règles) : 0.961 → 0.760 (0.816 sur les photos acceptées par la porte qualité) | il reste un écart sur le flou : lecteur plus robuste (§2, §6) ou photos plus nettes (porte qualité) |
| Écritures jamais vues | modèle entraîné sans les 5 polices du spécimen : 0.892 → 0.939 propre et 0.647 → 0.718 sev 2 grâce au « scripteur » synthétique de cette passe | le vrai risque terrain reste de vraies mains, pas des polices (§2, §3, §5) |
| Aucune photo du vrai carnet | les 5 photos réelles sont d'un autre modèle de registre (rejetées « page non reconnue ») | aucun chiffre sur le domaine réel |

## 1. Règle zéro : ce qui a le droit de sortir de la machine

Les consignes interdisent d'envoyer des données réelles de patientes à un service tiers. Donc, vers une API externe :

* **autorisé** : valeurs **fictives** générées par nos grammaires (stratégie 5), pages synthétiques, pages du spécimen
  public, images produites par l'API elle-même ;
* **interdit** : toute photo ou crop issu du terrain (même masqué), tout identifiant, tout export de la box.

C'est appliqué **dans le code** par `work/strat21/gemini_handwriting.py::Guard` : une valeur n'est envoyée que si ce
processus l'a tirée du générateur et qu'elle ne ressemble à aucun identifiant (CIN, téléphone, adresse —
`strat10.scan_text`) ; une image n'est renvoyée à l'API que si l'API l'a produite dans la session (empreinte SHA-256).
Les tests `work/strat21/test_gemini_handwriting.py` vérifient ces refus.

## 2. Gemini (modèle d'image) → banque d'écritures manuscrites réalistes — *implémenté, testé hors ligne, non exécuté*

**Pourquoi.** Nos 69 polices manuscrites + les déformations « scripteur » (inclinaison, espacement, ligne de base,
épaisseur, élastique, `synth_pages.writer_style`) restent des polices. Un modèle d'image génère des écritures de
mains bien plus variées. On ne lui envoie que des valeurs fictives (dates, TA, poids, « Normaux », « RAS »…).

**Comment** (une requête = une feuille de 12 lignes → ~10 échantillons gardés) :

```bash
pip install google-genai
set GEMINI_API_KEY=...                          # PowerShell : $env:GEMINI_API_KEY="..."
cd work/strat21
python gemini_handwriting.py --sheets 5 --dry_run                       # test du pipeline sans réseau
python gemini_handwriting.py --sheets 300 --model gemini-2.5-flash-image \
       --readback_model gemini-2.5-flash --out ~/dayone_local/hw_bank   # ≈ 3 000 lignes
```

Étapes : tirage de valeurs → prompt (stylo, profil de scripteur) → image → segmentation en lignes (profil
horizontal ; feuille rejetée si le nombre de lignes ne correspond pas) → **vérification** : on ne garde une ligne
que si notre CRNN y retrouve le texte demandé (CER ≤ 0.25 ou vraisemblance CTC ≥ −1.2 nat/caractère), et en option
si un modèle texte la relit à l'identique (les modèles d'image font des fautes) → patch d'encre RGBA → `bank.jsonl`.

**Utilisation pour l'entraînement** : `synth_page(..., bank=load_bank(dir), bank_p=0.3)` colle ces patches dans les
cases du formulaire (étiquette = texte de la banque) ; les crops passent ensuite par la dégradation photo et le vrai
recalage comme le reste :

```bash
python work/strat11/gen_dataset.py --pages 3000 --out ~/dayone_local/ds_bank --holdout 0 --hw_aug 0.6        --bank ~/dayone_local/hw_bank --bank_p 0.3
python work/strat11/train_crnn.py --data ~/dayone_local/ds_v1 ~/dayone_local/ds_v2 ~/dayone_local/ds_v3        ~/dayone_local/ds_bank --init models/crnn_final.pt --epochs 1 --bs 32 --lr 4e-4 --out ~/dayone_local/models/crnn_bank.pt
```

**Coût / effort.** 300 appels image (+ 300 relectures optionnelles). Vérifier le tarif courant du modèle d'image
(facturé à l'image) ; les noms de modèles évoluent : `--model` est paramétrable.
**Gain attendu (hypothèse).** Surtout sur les écritures jamais vues (0.85 → ?) ; à mesurer avec le protocole
« polices exclues » (`train_crnn.py --exclude_fonts …`), seul juge honnête.

## 3. Journée de collecte « fausses patientes » — la piste la plus rentable, sans API

**Pourquoi.** Le seul moyen de mesurer et d'apprendre le domaine réel (vraies mains, vrais stylos, vrais
téléphones, vrai papier du carnet) sans toucher à des données de patientes.

**Comment.**
1. Générer 50 dossiers fictifs complets (nos grammaires, ou les 200 lignes de `data/maternal_registry_synthetic.csv`)
   et imprimer une « feuille de dictée » par dossier.
2. Des sages-femmes recopient ces valeurs à la main dans des **carnets vierges** du vrai modèle, avec leurs stylos.
3. Elles photographient les pages avec leurs téléphones, dans leurs conditions (lumière, table, WhatsApp).
4. Les étiquettes sont connues exactement (la feuille de dictée) : aucune annotation, aucune API.

**Ce que ça débloque.** (a) un vrai jeu de test (le chiffre qui compte pour le jury et le terrain) ; (b) un
fine-tuning du CRNN sur le domaine réel ; (c) le gabarit du vrai carnet (`strat2/templates.py` depuis un scan du
carnet vierge) pour lire les photos aujourd'hui rejetées. Une variante avec Gemini (lecture-enseignant des pages
remplies) est possible car les données sont fictives, mais inutile : les étiquettes sont déjà connues.

## 4. Plafond de lisibilité des photos dégradées (Gemini Pro, données synthétiques seulement)

**Pourquoi.** À sev 2 une partie des crops est illisible même pour un humain (ex. page 11 : flou de bougé sur des
chiffres de 12 px). Savoir quelle part est *lisible* dit s'il reste du gain côté modèle ou s'il faut surtout
durcir la porte qualité (re-photo).

**Comment.** Envoyer à `gemini-2.5-pro` les crops **synthétiques** (`~/dayone_local/ds_v3`, `sev` ≥ 2) avec l'étiquette
de champ, comparer sa lecture à la vérité. Les crops synthétiques ne contiennent aucune donnée réelle ; le Guard
doit être étendu avec une liste blanche « répertoires de données synthétiques » avant d'autoriser ces envois.
**Décision attendue.** Si Gemini Pro fait ≫ 0.70 sur ces crops : investir dans le modèle (§2, §6). Sinon : la porte
qualité est le bon levier (déjà en place : variance du laplacien, 90 % des photos mauvaises rejetées pour 12 % de faux rejets).

## 5. Corpus publics d'écriture manuscrite (inscription requise)

| Corpus | Langue | Accès | Usage |
|---|---|---|---|
| RIMES | français (courriers manuscrits) | inscription, usage recherche | pré-entraînement du CRNN sur de vraies mains françaises |
| IAM | anglais | inscription, non commercial | idem |
| KHATT / IFN-ENIT | arabe | inscription | partie arabe (aujourd'hui mesurée sur synthétique seulement) |
| MADBase / AHDBase | chiffres arabes-indiens | téléchargement | chiffres ٠-٩ |

Procédure : pré-entraîner le CRNN (même alphabet, `train_crnn.py --data …`) sur ces lignes, puis fine-tuning sur nos
crops synthétiques (`--init`). Vérifier la licence avant tout usage hors recherche.

## 6. GPU plus gros (cloud ou box avec 24 Go) — modèles plus lourds en local

* **Lecteur plus large** : notre CRNN fait 6.7 M paramètres, contraint par 8 Go et ~1 h d'entraînement. Un CRNN 3×
  plus large ou un TrOCR-small/base fine-tuné sur nos crops est réaliste sur un GPU 24 Go (Kaggle/Colab : seulement
  des données synthétiques). Le modèle final reste hébergé sur la box, hors ligne.
* **Second lecteur VLM sur la box** : Qwen2.5-VL-3B (4 bits) faisait 0.65 vs 0.84 pour le CRNN sur nos crops et n'a
  pas été retenu. Un 7B/32B quantifié (≥ 16–24 Go) pourrait aider sur le texte libre long (noms d'établissement,
  commentaires), uniquement pour les champs `À_RÉVISER`. Tout reste dans l'établissement : pas de tiers.

## 7. WhatsApp réel (compte Meta)

L'adaptateur `work/strat14/whatsapp_adapter.py` (webhook, boutons interactifs, idempotence) est testé hors ligne.
Pour une démonstration réelle : numéro de test WhatsApp Cloud API (Meta for Developers), `WHATSAPP_TOKEN`,
`PHONE_NUMBER_ID`, URL publique du webhook. **Uniquement avec des données fictives** : passer des photos de
registres réels par Meta enfreindrait la règle « pas de tiers » ; en production le canal reste la PWA locale.

## Récapitulatif

| Piste | Besoin externe | Effort | Gain visé | Statut |
|---|---|---|---|---|
| §3 collecte « fausses patientes » | carnets vierges + sages-femmes | 1 journée | domaine réel : test + entraînement + gabarit du vrai carnet | protocole prêt |
| §2 banque Gemini | clé API | ~1 h + coût API | écritures jamais vues | code + tests prêts (`strat21`) |
| §4 plafond de lisibilité | clé API | 1 h | décider modèle vs porte qualité | protocole |
| §5 corpus publics | inscriptions | 1–2 jours | vraies mains FR/EN/AR | protocole |
| §6 GPU 24 Go | cloud/box | 1 jour | lecteur plus robuste au flou | protocole |
| §7 WhatsApp | compte Meta | 0.5 jour | démo du canal réel | adaptateur prêt |
