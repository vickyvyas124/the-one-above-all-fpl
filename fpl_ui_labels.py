"""Pure presentation labels shared by the FPL sidebar and status strip."""


def status_chip_labels(auto_controls, latest_bank_known, auto_ft_confident):
    """Return the bank and free-transfer source labels for the current controls."""
    if not auto_controls:
        return "manual", "manual"

    bank_label = "synced" if latest_bank_known else "fallback estimate"
    free_transfer_label = "history-derived" if auto_ft_confident else "estimated"
    return bank_label, free_transfer_label
