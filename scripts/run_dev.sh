#!/bin/bash

# Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0

uvicorn perturber.api:app --host 0.0.0.0 --port 8001 --reload
