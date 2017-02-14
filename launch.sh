#!/bin/bash

u1=$1
i1='input.tmp'
d1='data.tmp'
touch data.tmp
touch input.tmp

cat $1 > $i1
#i1=`cat $u1 > $i1`

python3 MPQuery-2.py > $d1
mv $d1 data/$d1

head $d1 -n 20
touch $d1


