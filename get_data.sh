#!/usr/bin/env bash
cd data/
curl "https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip" -o "fb15k-237.zip"
unzip "fb15k-237.zip"
rm "fb15k-237.zip"
mv "Release" "fb15k-237"
rm "fb15k-237/text_cvsc.txt" "fb15k-237/text_emnlp.txt"

cd ..
curl "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz" -o "wn18.tar"
tar -xf "wn18.tar"
rm "wn18.tar"
mv "wordnet-mlj12" "wn18"
cd wn18/
mv "wordnet-mlj12-train.txt" "train.txt"
mv "wordnet-mlj12-test.txt" "test.txt"
mv "wordnet-mlj12-valid.txt" "valid.txt"
mv "wordnet-mlj12-definitions.txt" "definitions.txt"
