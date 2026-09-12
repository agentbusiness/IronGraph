#!/usr/bin/env bash
# Fetch the open datasets that back the IronGraph Cypher reference examples.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p raw
get() { # get <url> <filename>
  if [ -s "raw/$2" ]; then echo "have   $2"; return 0; fi
  if curl -sSL -m 900 -o "raw/$2.part" "$1"; then mv "raw/$2.part" "raw/$2"; echo "got    $2 ($(du -h "raw/$2" | cut -f1))";
  else echo "FAILED $2 <- $1"; rm -f "raw/$2.part"; fi
}
S=https://snap.stanford.edu/data
get $S/soc-sign-bitcoinotc.csv.gz            soc-sign-bitcoinotc.csv.gz
get $S/soc-sign-bitcoinalpha.csv.gz          soc-sign-bitcoinalpha.csv.gz
get $S/soc-Epinions1.txt.gz                  soc-Epinions1.txt.gz
get $S/bigdata/communities/com-dblp.ungraph.txt.gz     com-dblp.ungraph.txt.gz
get $S/bigdata/communities/com-dblp.top5000.cmty.txt.gz com-dblp.top5000.cmty.txt.gz
get $S/cit-HepTh.txt.gz                      cit-HepTh.txt.gz
get $S/cit-HepTh-abstracts.tar.gz            cit-HepTh-abstracts.tar.gz
get $S/cit-HepTh-dates.txt.gz                cit-HepTh-dates.txt.gz
get $S/facebook_combined.txt.gz              facebook_combined.txt.gz
get $S/email-Eu-core.txt.gz                  email-Eu-core.txt.gz
get $S/email-Eu-core-department-labels.txt.gz email-Eu-core-department-labels.txt.gz
get $S/sx-mathoverflow.txt.gz                sx-mathoverflow.txt.gz
get $S/sx-askubuntu.txt.gz                   sx-askubuntu.txt.gz
get $S/CollegeMsg.txt.gz                     CollegeMsg.txt.gz
O=https://raw.githubusercontent.com/jpatokal/openflights/master/data
get $O/airports.dat  airports.dat
get $O/routes.dat    routes.dat
get $O/airlines.dat  airlines.dat
