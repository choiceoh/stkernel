# onepass26 refused before GPU hold

The normal freshness guard rejected source `055914aeb719c1769e05cdb863e43a88b2ee47af` before admission: required main `396675e62d5734e07412624f2b1a53778adef167` startup/audit changes were absent. The launch response is `accepted=false`, startup-failed, return code3, and has no accepted ticket. The exact refusal log explicitly says `no GPU hold`.

No canonical measurement record or GPU26/CPU26 result is invented. Source25 and original CPU24 reuse provenance remain their original identities. The new matched B/A on source27 is separate work. This archive retains original submit/config/job/startup bytes and missing paths; it does not attribute a performance or numerical failure to the candidate.
