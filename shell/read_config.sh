#!/bin/bash
# set -e
 
exit_script(){
   exit 1
}

if [ "$#" = 0 ]; then
    echo "参数错误，命令格式为:    ./read_config.sh configfile"
    exit_script
else
    configPath=$1
fi

function get_line_num(){
    local configKey=$1
    grep -n -E '^\[' ${configPath} |grep -A 1 "\[${configKey}\]"|awk -F ':' '{print $1}'|xargs
}
 
function get_config(){
    #local configPath=$1
    local configKey=$1
    local configName=$2
    local line_num=$(get_line_num $configKey)
    local startLine=$(echo $line_num |awk '{print $1}')
    local endLine=$(echo $line_num|awk '{print $2}')
    if [ ${endLine} ];then
        sed -n "${startLine},${endLine} s/${configName}=//p" ${configPath}
    else
        sed -n "${startLine},$ s/${configName}=//p" ${configPath}
    fi
}
 
if [ -f $configPath ];then
    DEPLOY_HOST=$(get_config 220 ip)
else
    echo ${configPath}"文件不存在,请检查配置文件是否存在"
    exit_script
fi

DEPLOY_CAMERA=$(get_config 220 camera)
# 设置的相机，转为数组形式
cameras=(`echo $DEPLOY_CAMERA | tr ',' ' '` )
echo ${DEPLOY_HOST}
echo ${DEPLOY_CAMERA}


isIN=false
for var in ${cameras[@]}
do
    echo $var
    if [ "$var" = "47" ]; then
        isIN=true
    fi
done
echo $isIN

cur_host_ip=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}') #|tr -d "addr:"

echo $cur_host_ip

