"""Polarity-channel guard unit tests (text-level, no models).

The NLI polarity gate over-fires on legal/deontic text (ContractNLI FPs
849/604/1829, 2026-10-12): negation with an exception clause, negation of a
deontic requirement, and antonymy across different subjects are NOT
contradictions of the claim.  The guards make the channel conservative in
exactly those three shapes while leaving plain negation conflicts live.
"""

from __future__ import annotations

import unittest

from hufu.engine.reference_check import (
    _deontic_scope_negation,
    _exception_scoped_negation,
)


class ExceptionScopeGuardTests(unittest.TestCase):
    def test_negation_with_exception_clause_is_scoped(self) -> None:
        # ContractNLI 604: "shall not make any copies ... except as required
        # for the Permitted Purpose" — the exception scopes the negation.
        text = ("The Receiving Party shall not make any copies, reproductions "
                "or abstracts of the Confidential Information of the Disclosing "
                "Party except as specifically may be required for the Permitted Purpose.")
        self.assertTrue(_exception_scoped_negation(text))

    def test_unless_and_other_than_markers(self) -> None:
        self.assertTrue(_exception_scoped_negation(
            "The tenant shall not sublet the premises unless the landlord consents in writing."))
        self.assertTrue(_exception_scoped_negation(
            "No party may disclose the data other than to authorized affiliates."))

    def test_plain_negation_not_scoped(self) -> None:
        # Golden declarative negation conflicts must keep firing.
        self.assertFalse(_exception_scoped_negation("The adapter does not support USB-C."))
        self.assertFalse(_exception_scoped_negation(
            "The shop is open on weekends except the display is new."))  # no negation


class DeonticScopeGuardTests(unittest.TestCase):
    def test_not_required_is_not_prohibition(self) -> None:
        # ContractNLI 849: "may, but is not required to, mark ..." negates
        # the requirement, not the act.
        self.assertTrue(_deontic_scope_negation(
            'For convenience, the Disclosing Party may, but is not required to, '
            'mark written Confidential Information with the legend "Confidential".'))

    def test_other_deontic_forms(self) -> None:
        self.assertTrue(_deontic_scope_negation("The vendor is not obligated to provide updates."))
        self.assertTrue(_deontic_scope_negation("No party is under any obligation to renew."))

    def test_plain_prohibition_not_deontic_scope(self) -> None:
        self.assertFalse(_deontic_scope_negation("The Receiving Party shall not disclose the data."))
        self.assertFalse(_deontic_scope_negation("The adapter does not support USB-C."))


class PolarityGuardIntegrationTests(unittest.TestCase):
    """Model-backed: the three ContractNLI FPs must flip to consistent, and
    plain negation contradictions must still fire.  Gated by HUFU_RUN_E2E."""

    def setUp(self) -> None:
        import os
        if os.environ.get("HUFU_RUN_E2E") != "1":
            self.skipTest("requires HUFU_RUN_E2E=1")

    def _check(self, hyp: str, prem: str):
        from hufu import check_pair

        return check_pair(hyp, prem, engine="reference",
                          query_is_question=False, distance_threshold=-1.0)

    def test_contractnli_604_exception_clause(self) -> None:
        v = self._check(
            "Receiving Party may create a copy of some Confidential Information in some circumstances.",
            "The Receiving Party shall not make any copies, reproductions or abstracts of the "
            "Confidential Information of the Disclosing Party except as specifically may be "
            "required for the Permitted Purpose.")
        self.assertTrue(v.consistent, v.evidence)

    def test_contractnli_849_not_required(self) -> None:
        v = self._check(
            "All Confidential Information shall be expressly identified by the Disclosing Party.",
            'For convenience, the Disclosing Party may, but is not required to, mark written '
            'Confidential Information with the legend "Confidential" or an equivalent designation.')
        self.assertTrue(v.consistent, v.evidence)

    def test_contractnli_1829_cross_subject_antonymy(self) -> None:
        v = self._check(
            "Some obligations of Agreement may survive termination of Agreement.",
            "9. This Agreement shall expire and cease to have any force or effect on the earlier "
            "of (i) eighteen months from the date hereof or (ii) the date of consummation of a "
            "transaction between the parties.")
        self.assertTrue(v.consistent, v.evidence)

    def test_plain_negation_still_contradiction(self) -> None:
        # Recall control: guards must not swallow a direct negation conflict.
        v = self._check(
            "The adapter supports USB-C charging.",
            "The adapter does not support USB-C charging.")
        self.assertFalse(v.consistent, v.evidence)

    def test_contested_subject_slot_still_contradiction(self) -> None:
        # golden250 M14: shared predicate + differing subject = the subject
        # IS the contested slot; the cross-subject guard must not veto.
        v = self._check(
            "Jane Austen wrote the novel Emma.",
            "Charlotte Brontë wrote the novel Emma.")
        self.assertFalse(v.consistent, v.evidence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
