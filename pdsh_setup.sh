hang_dir=/apdcephfs_sh8/share_301266059/stephenruan/code/hang
cd $hang_dir
bash get_pssh_hosts.sh
pssh_hosts=${hang_dir}/pssh.hosts
EXP_IP_LIST=$(grep -vE '^(#.*|$)' "${pssh_hosts}" | paste -sd, -)
echo ${EXP_IP_LIST}
# NOW_DIR="$PWD"
NOW_DIR="/apdcephfs_cq8/share_1611098/ruanzheng/code/src/R1-V"
pdsh -w $EXP_IP_LIST "cd $NOW_DIR; bash setup.sh"
