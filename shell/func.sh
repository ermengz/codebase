#!/bin/bash
# GNU bash, version 3.2.57(1)-release (x86_64-apple-darwin21)

echo $*
echo "$*" $*
echo "$@" $@
# echo $1

# for 循环
echo "for loop test"
for di in "$@"; do
    echo ${di}
done
# output
# 1
# 2
# 3

for di in "$*"; do
    echo ${di}
done
# output
# 1 2 3

# 函数调用
# foo (){
#     local p1=$1
#     local p2=$2
#     local p3=$3
#     echo ${p1}
#     echo ${p2}
#     echo ${p3}
# }
# foo 1 2 3




