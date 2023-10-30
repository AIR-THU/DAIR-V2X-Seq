## V2X-Seq TODO: downloading datasets sucessfully

#! /bin/bash

fileid='1gjOmGEBMcipvDzu2zOrO9ex_OscUZMYY'
filename='V2X-Seq-SPD-Example.zip'
cd ./dataset/v2x-seq-spd
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
unzip ${filename}
rm ${filename}

fileid='1vV2BZvBWkum-j0r82JOjAajlSWB7kyU2'
filename='V2X-Seq-TFD-Example.zip'
cd ../v2x-seq-tfd
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
unzip ${filename}
rm ${filename}

cd ../..