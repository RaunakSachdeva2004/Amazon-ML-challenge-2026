# Methodology Write-up

## Normalization
Applied Unicode NFKC, lowercasing, and specific rules to standardize suffixes (e.g., corp, ltd) and address abbreviations (e.g., st, rd). Regex was used to extract PIN/ZIP codes and street numbers.

## Blocking
Strategies:
1. Token Inverted Index
2. TF-IDF KNN on business name
3. TF-IDF KNN on business address
4. Sorted Neighborhood on name prefix
5. Phonetic blocking (Metaphone)

Top 60 candidates were kept per source entity after aggregating scores across strategies.

## Features
Extracted features such as Levenshtein distance (normalized), Jaro-Winkler, Jaccard, Dice similarity, token sort/set ratio, acronym match, longest common subsequence (LCS) length norm, absolute/relative string length differences, suffix-stripped exact matches, and address-specific matches (postal code, street number).

## Model
A LightGBM binary classifier (objective='binary') was trained on the constructed features. It complies with the <=8B parameter constraint and MIT license requirements.

## Decoding
Tuned the prediction threshold over the validation set to maximize the macro F0.5 score.

## Known Failure Modes
- Difficulties with extremely generic names without sufficient address details.
- Sparse blocks where spelling differences evade all 5 blocking strategies.
