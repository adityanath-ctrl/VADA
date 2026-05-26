APP=main:app
HOST=0.0.0.0
PORT=8000

CERT_DIR=.cert
KEY=$(CERT_DIR)/key.pem
CERT=$(CERT_DIR)/cert.pem

.PHONY: run cert clean kill

# 🔐 Generate SSL certificates
# cert:
# 	mkdir -p $(CERT_DIR)
# 	openssl req -x509 -newkey rsa:2048 -nodes \
# 	-keyout $(KEY) \
# 	-out $(CERT) \
# 	-days 365 \
# 	-subj "/CN=localhost"

run:
	uvicorn $(APP) \
	--host $(HOST) \
	--port $(PORT) \
	--ssl-keyfile $(KEY) \
	--ssl-certfile $(CERT)

kill:
	lsof -t -i:8000 | xargs kill -9 || true

clean:
	rm -rf $(CERT_DIR)