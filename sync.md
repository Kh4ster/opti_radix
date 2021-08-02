# To add upstream

git clone git@github.com:Kh4ster/cuda_bench_template.git New_Repo
cd New_Repo
git remote set-url origin https://github.com/userName/New_Repo
git remote add upstream git@github.com:Kh4ster/cuda_bench_template.git
git push origin master

# To sync with template :

git fetch upstream
git merge upstream/master
git push origin master