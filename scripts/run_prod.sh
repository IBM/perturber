#!/bin/bash

uvicorn perturber.api:app --host 0.0.0.0 --port 8080
