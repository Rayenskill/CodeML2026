# Record lifecycle

Every page photographed by the midwife is one **record**. Its state lives encrypted on the phone (IndexedDB +
AES-GCM, key derived from the device PIN) and every change is appended to an event log. There is **one
transition table**, `work/strat15/app/lifecycle.json`, loaded by the phone app and by `work/strat8/offline.py`,
whose property-based tests (network cuts, server errors, duplicate acks, crashes) check that no record is ever lost
and no illegal transition happens.

```mermaid
stateDiagram-v2
    [*] --> CAPTURÉ: photo accepted by the on-device quality gate
    CAPTURÉ --> EN_ATTENTE_IA: encrypted & queued (works offline)
    CAPTURÉ --> DOUBLON_SUSPECTÉ: same photo already in the session
    DOUBLON_SUSPECTÉ --> EN_ATTENTE_IA: midwife keeps it
    TRAITÉ_IA --> DOUBLON_SUSPECTÉ: same content as a page already read
    DOUBLON_SUSPECTÉ --> À_RÉVISER
    DOUBLON_SUSPECTÉ --> VALIDÉ
    EN_ATTENTE_IA --> TRAITÉ_IA: edge box read the page
    EN_ATTENTE_IA --> ÉCHEC_TRAITEMENT: box error / timeout
    ÉCHEC_TRAITEMENT --> EN_ATTENTE_IA: automatic retry (back-off)
    EN_ATTENTE_IA --> RÉVISION_MANUELLE_REQUISE: AI unavailable → manual entry
    ÉCHEC_TRAITEMENT --> RÉVISION_MANUELLE_REQUISE: permanent error (4xx) or 5 failures
    TRAITÉ_IA --> VALIDÉ: every field confident
    TRAITÉ_IA --> À_RÉVISER: uncertain fields or page not recognised
    À_RÉVISER --> VALIDÉ: midwife confirms / corrects
    À_RÉVISER --> CAPTURÉ: "Reprendre la photo" (new version of the same record)
    À_RÉVISER --> RÉVISION_MANUELLE_REQUISE: page not recognised → manual entry
    RÉVISION_MANUELLE_REQUISE --> VALIDÉ: manual entry finished / match retried
    RÉVISION_MANUELLE_REQUISE --> CAPTURÉ: retake after a failure
    VALIDÉ --> PATIENTE_LIÉE: midwife picks the patient (code-based candidates)
    VALIDÉ --> RÉVISION_MANUELLE_REQUISE: "Je ne sais pas" (nothing linked, nothing created)
    PATIENTE_LIÉE --> ENREGISTRÉ: booklet closed (re-scan diff resolved)
    ENREGISTRÉ --> SYNCHRONISÉ: central store acknowledged (idempotent key record:version)
    ENREGISTRÉ --> ÉCHEC_SYNCHRO: network / server error
    ÉCHEC_SYNCHRO --> ENREGISTRÉ: retry when connectivity returns
    SYNCHRONISÉ --> [*]
```

| State | Where | Meaning |
|---|---|---|
| CAPTURÉ | phone | photo stored encrypted, not yet queued |
| EN_ATTENTE_IA | phone | waiting for the edge box (offline or busy); retried with exponential back-off |
| TRAITÉ_IA | phone | extraction received (values, statuses, confidences, evidence crops) |
| À_RÉVISER | phone | the agent asks the midwife about uncertain fields (≤ 8 questions in a row) |
| VALIDÉ | phone | every field confirmed, corrected, or explicitly left "à réviser" |
| PATIENTE_LIÉE | phone + box registry | linked to a patient profile by the registry code; the midwife decides |
| ENREGISTRÉ | phone | ready to sync (in the outbox) |
| SYNCHRONISÉ | central store | acknowledged once; duplicates are ignored server-side |
| ÉCHEC_TRAITEMENT / ÉCHEC_SYNCHRO | phone | failures, retried automatically, visible in the queue |
| DOUBLON_SUSPECTÉ | phone | same bytes already captured in the session |
| RÉVISION_MANUELLE_REQUISE | phone | manual entry (AI unavailable / page not recognised) or undecided match |

**Nothing stays stuck after a restart.** At unlock the app resumes every non-terminal state: a requested retake
(`CAPTURÉ`, persisted flag), an undecided duplicate (asked again), pages linked but not yet registered
(`PATIENTE_LIÉE`: the registration finishes when the box is reachable; the re-scan diff is optional), an undecided
match (offered again) and pages waiting for manual entry. A page the box rejects permanently (e.g. unreadable
image) goes to manual review instead of being retried forever.

**Field statuses** (per field, separate from the record state): `CONNU`, `INCONNU` (written "?", "inconnu", "NSP"),
`NON_FOURNI` (empty box or written dash), `ILLISIBLE` (ink present, unreadable), `NON_APPLICABLE` (blank and excluded by
the form's own logic, e.g. caesarean indication after a vaginal delivery), `À_RÉVISER` (read but doubtful).
