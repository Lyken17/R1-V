export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

source $(conda info --base)/etc/profile.d/conda.sh
# conda env remove -n r1-v -y

conda env list