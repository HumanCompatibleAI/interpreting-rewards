#!/usr/bin/env bash

aws s3 cp s3://interpreting-rewards/eric-dev/agents agents --recursive
aws s3 cp s3://interpreting-rewards/eric-dev/videos videos --recursive
aws s3 cp s3://interpreting-rewards/eric-dev/reward-models reward-models --recursive




