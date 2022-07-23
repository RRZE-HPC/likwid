#!/bin/bash

GITHUB_ORG="RRZE-HPC"
GITHUB_REPO="likwid"
GITHUB_SHA="${CI_COMMIT_SHA}"

cat << EOF > headers.curl
Accept: application/vnd.github+json
Authorization: token ${GITHUB_API_TOKEN}
EOF
#cat << EOF > success.json
#{
#  "state" : "success",
#  "target_url" : "${CI_PIPELINE_URL}",
#  "description" : "CI runs at NHR@FAU systems successful",
#  "context" : "continuous-integration/gitlab"
#}
#EOF
cat << EOF > success.json
{
  "state" : "success",
  "target_url" : "${CI_PIPELINE_URL}",
  "description" : "CI runs at NHR@FAU systems successful"
}
EOF
cat << EOF > failure.json
{
  "state" : "failure",
  "target_url" : "${CI_PIPELINE_URL}",
  "description" : "CI runs at NHR@FAU systems failed"
}
EOF
cat << EOF > pending.json
{
  "state" : "pending",
  "target_url" : "${CI_PIPELINE_URL}",
  "description" : "CI runs at NHR@FAU systems pending"
}
EOF
GITHUB_API_URL="https://api.github.com/repos/${GITHUB_ORG}/${GITHUB_REPO}/statuses/${GITHUB_SHA}"
if [ "$1" == "success" ]; then
	cat success.json
	curl -s -X POST -H @headers.curl "${GITHUB_API_URL}" -d @success.json
elif [ "$1" == "failure" ]; then
	cat failure.json
	curl -s -X POST -H @headers.curl "${GITHUB_API_URL}" -d @failure.json
elif [ "$1" == "pending" ]; then
	cat pending.json
	curl -s -X POST -H @headers.curl "${GITHUB_API_URL}" -d @pending.json
fi
