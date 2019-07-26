#!/usr/bin/env bash
cd data/
curl "https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip" -o "fb15k-237.zip"
unzip "fb15k-237.zip"
rm "fb15k-237.zip"
mv "Release" "fb15k-237"
rm "fb15k-237/text_cvsc.txt" "fb15k-237/text_emnlp.txt"


curl "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz" -o "wn18.tar"
tar -xf "wn18.tar"
rm "wn18.tar"
mv "wordnet-mlj12" "wn18"
mv "wn18/wordnet-mlj12-train.txt" "wn18/train.txt"
mv "wn18/wordnet-mlj12-test.txt" "wn18/test.txt"
mv "wn18/wordnet-mlj12-valid.txt" "wn18/valid.txt"
mv "wn18/wordnet-mlj12-definitions.txt" "wn18/definitions.txt"


curl 'https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz' -o 'fb15.tar';
tar -xf "fb15.tar"
mv "FB15k" "fb15k"
rm "fb15.tar"
mv "fb15k/freebase_mtr100_mte100-train.txt" "fb15k/train.txt"
mv "fb15k/freebase_mtr100_mte100-test.txt" "fb15k/test.txt"
mv "fb15k/freebase_mtr100_mte100-valid.txt" "fb15k/valid.txt"



