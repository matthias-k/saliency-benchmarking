NAME=benchmarking
imagename=matthias-k/mit-saliency-benchmark
cmd=/bin/bash

run: build-docker
	#agmb-docker run --rm -it -w `pwd` --name=$(NAME) matthiask/mit-saliency-benchmark /bin/bash
	docker run --rm -it --user root -e NB_UID=$(shell id -u) -e NB_GID=$(shell id -g) -e NB_USER=$(USER) -v $(HOME):$(HOME) -w $(shell pwd) -v /usr/local/MATLAB/R2018b:/usr/local/MATLAB/R2018b $(imagename) start.sh $(cmd)

build-docker:
	cd docker/matlab && make build

nginxargs='-d'
run-nginx:
	docker rm -f mkuemmerer-nginx | true
	docker run --rm $(nginxargs) --name mkuemmerer-nginx -v $(shell pwd)/html:/usr/share/nginx/html -p 4242:80 nginx

update-website:
	cd html && git commit -asm "Updates" && git push
