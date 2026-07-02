.PHONY: install-python-package setup clean create-directories iopaint-server benchmark noti-setup noti-test

# ตั้งค่าตัวแปร
PYTHON := python3
VENV := .venv
V_PIP := $(VENV)/bin/pip
V_PYTHON := $(VENV)/bin/python3
V_IOPAINT := $(VENV)/bin/iopaint

# vermeer: root NVMe เกือบเต็ม — ชี้ cache ไป HDD ที่ mount แล้ว เช่น
# make setup CACHE_DIR=/mnt/hdd/model-cache
CACHE_DIR ?=
ifneq ($(CACHE_DIR),)
CACHE_ENV := HF_HOME=$(CACHE_DIR)/huggingface TORCH_HOME=$(CACHE_DIR)/torch
else
CACHE_ENV :=
endif

install-python-package:
	@echo "Installing requirements python package"

	# 1. สร้าง venv ก่อนเป็นอันดับแรก เพื่อเลี่ยงปัญหา PEP 668
	@if sudo -n true 2>/dev/null; then \
		sudo apt-get update && sudo apt-get install -y libjpeg-dev zlib1g-dev libpng-dev; \
	else \
		echo "Skipping apt-get (no sudo — assuming HPC)"; \
	fi
	@$(PYTHON) -m venv $(VENV)
	@$(V_PYTHON) -m pip install --upgrade pip

	# vermeer (RTX 3090)
	@$(V_PYTHON) -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
	# OMEN (RTX 2070, WSL2)
	# @$(V_PYTHON) -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

	# 2. Core pipeline: IOPaint (LaMa/MI-GAN/SD + SAM plugin) + benchmark deps
	@$(V_PIP) install iopaint
	@$(V_PIP) install opencv-python matplotlib scikit-image diffusers transformers accelerate
	# metrics: LPIPS (synthetic cases), BRISQUE/NIQE via pyiqa
	@$(V_PIP) install lpips pyiqa

	# 3. SAM 2 สำหรับ Stage 1 mask generation (custom path)
	@echo "Installing SAM 2..."
	@$(V_PIP) install git+https://github.com/facebookresearch/sam2.git

	# 4. requirements.txt เพิ่มเติม (ถ้ามี)
	@if [ -f requirements.txt ]; then $(V_PIP) install -r requirements.txt; fi

	# optional: YOLOv8-seg สำหรับ auto-masking ตาข่ายแบบ fixed pattern
	# @$(V_PIP) install ultralytics

clean:
	@echo "Cleaning up"
	@rm -rf $(VENV)

setup: create-directories install-python-package
	@echo "ALL requirements are installed"

create-directories:
	@mkdir -p data masks methods results

# Stage 2 fast path: IOPaint server (LaMa) + SAM plugin สำหรับ interactive masking
iopaint-server:
	@$(CACHE_ENV) $(V_IOPAINT) start --model=lama --device=cuda --port=8080 \
		--enable-interactive-seg --interactive-seg-device=cuda

# รัน benchmark ทั้ง pipeline → figures 16:9 ใน results/
benchmark:
	@$(CACHE_ENV) $(V_PYTHON) run_benchmark.py

# noti-setup:
# 	@echo "Setting up telegram-noti..."
# 	@git submodule update --init
# 	@if [ ! -f telegram-noti/.env ]; then \
# 		cp telegram-noti/.env.example telegram-noti/.env; \
# 		echo "Created telegram-noti/.env — fill in your token and chat ID, then run: make noti-test"; \
# 	else \
# 		echo "telegram-noti/.env already exists"; \
# 	fi

# noti-test:
# 	@cd telegram-noti && $(V_PYTHON) setup.py
