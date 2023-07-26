#!/bin/bash
PMCXCL_BUILD_VERSION=$(awk -F"-" '{ print $2 }' <<< $(ls dist/ | head -1))
PMCXCL_VERSIONS_STRING=$(pip index versions pmcxcl | grep versions:)
PMCXCL_VERSIONS_STRING=${PMCXCL_VERSIONS_STRING#*:}
UPLOAD_TO_PYPI=1
while IFS=', ' read -ra PMCXCL_VERSIONS_ARRAY; do
  for VERSION in "${PMCXCL_VERSIONS_ARRAY[@]}"; do
    if [ "$PMCXCL_BUILD_VERSION" = "$VERSION" ]; then
      UPLOAD_TO_PYPI=0
    fi
  done;
done <<< "$PMCXCL_VERSIONS_STRING"
if [ "$UPLOAD_TO_PYPI" = 1 ]; then
  echo "Wheel version wasn't found on PyPi.";
else
  echo "Wheel was found on PyPi.";
fi
echo "perform_pypi_upload=$UPLOAD_TO_PYPI" >> $GITHUB_OUTPUT
