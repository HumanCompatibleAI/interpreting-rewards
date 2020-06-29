#!/usr/bin/env bash

aws s3 cp agents s3://interpreting-rewards/eric-dev/agents --recursive
aws s3 cp videos s3://interpreting-rewards/eric-dev/videos --recursive
aws s3 cp reward-models s3://interpreting-rewards/eric-dev/reward-models --recursive




