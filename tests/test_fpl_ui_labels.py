import unittest

from fpl_ui_labels import status_chip_labels


class StatusChipLabelTests(unittest.TestCase):
    def test_manual_override_labels_both_values_manual(self):
        self.assertEqual(status_chip_labels(False, True, True), ("manual", "manual"))
        self.assertEqual(status_chip_labels(False, False, False), ("manual", "manual"))

    def test_auto_sync_labels_known_bank_and_confident_transfers(self):
        self.assertEqual(status_chip_labels(True, True, True), ("synced", "history-derived"))

    def test_auto_sync_labels_fallback_bank_and_confident_transfers(self):
        self.assertEqual(status_chip_labels(True, False, True), ("fallback estimate", "history-derived"))

    def test_auto_sync_labels_known_bank_and_estimated_transfers(self):
        self.assertEqual(status_chip_labels(True, True, False), ("synced", "estimated"))

    def test_auto_sync_labels_fallback_bank_and_estimated_transfers(self):
        self.assertEqual(status_chip_labels(True, False, False), ("fallback estimate", "estimated"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
