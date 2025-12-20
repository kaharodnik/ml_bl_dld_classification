######This is an example of preprocessing pipeline, with a few examples of steps used for
###cleaning the data and extracting the features. 

#!/bin/bash
set -euo pipefail

DIR="C:/Users/Katya/Desktop/Dissertation"
CHA_DIR="$DIR/DATA_FILES/SLI/7-9"
POS_DIR="$DIR/Data_Processing/my_stem/ML_SLI_combined/SLI_4-6"
OUT_DIR="$DIR/outputs"

mkdir -p "$OUT_DIR"

#######################################
# 1. Extract CHI utterances + tokenize
#######################################
extract_and_tokenize() {
  for file in "$CHA_DIR"/*.cha; do
    base=$(basename "$file" .cha)

    grep '^\*CHI' "$file" \
      | sed 's/[*CHI:\t]//g' \
      | sed 's/\[.*\]//g' \
      | tr -sc 'A-Za-z~' '\n' \
      > "$OUT_DIR/$base.tr"
  done
}

####################################################
# 2. Word count per file - total number of words - TNW
####################################################
word_count() {
  wc -l "$OUT_DIR"/*.tr > "$OUT_DIR/word_counts.txt"
}

#######################################
# 3. NDW - extracting unique words
#######################################
ndw() {
  for file in "$OUT_DIR"/*.tr; do
    sort "$file" | uniq -c > "$file.freq"
  done
}

###################################################################################
# 4. Extract POS tags (no lemmas) from the output of morphological analyzer MyStem
##################################################################################
extract_pos_sequences() {
  for file in "$POS_DIR"/*.pos; do
    sed 's/.*=//' "$file" \
      | cut -d '|' -f1 \
      | tr -d '}' \
      | tr ',' '_' \
      | tr '\n' ' ' \
      | tr -s ' ' \
      > "$file.POS"
  done
}

#######################################
# 5. Extract verb & noun unique lemmas
#######################################
extract_lemmas() {
  for file in "$POS_DIR"/*.pos; do
    base=$(basename "$file")

    grep '=V,' "$file" \
      | cut -d '{' -f2 \
      | cut -d '=' -f1 \
      | tr -d '?' \
      | sort | uniq -c \
      > "$OUT_DIR/$base.verbs.txt"

    grep '=S,' "$file" \
      | cut -d '{' -f2 \
      | cut -d '=' -f1 \
      | tr -d '?' \
      | sort | uniq -c \
      > "$OUT_DIR/$base.nouns.txt"
  done
}

#######################################
# Run everything
#######################################
extract_and_tokenize
word_count
ndw
extract_pos_sequences
extract_lemmas

echo "Pipeline completed successfully."
