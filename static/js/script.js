/**
 * SpendWise AI - Main Interactive Client Scripts
 */

document.addEventListener("DOMContentLoaded", function () {
  // 1. Mobile Sidebar Toggle
  const mobileToggle = document.getElementById("mobileToggle");
  const sidebar = document.getElementById("sidebar");

  if (mobileToggle && sidebar) {
    mobileToggle.addEventListener("click", function () {
      sidebar.classList.toggle("open");
    });

    // Close when clicking outside on mobile
    document.addEventListener("click", function (e) {
      if (
        window.innerWidth <= 768 &&
        !sidebar.contains(e.target) &&
        !mobileToggle.contains(e.target) &&
        sidebar.classList.contains("open")
      ) {
        sidebar.classList.remove("open");
      }
    });
  }

  // 2. Alert Dismissal
  const closeButtons = document.querySelectorAll(".alert-close");
  closeButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      const alert = this.closest(".alert");
      if (alert) {
        alert.style.transition = "opacity 0.3s ease";
        alert.style.opacity = "0";
        setTimeout(() => alert.remove(), 300);
      }
    });
  });
});

// Modal Utilities
function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.add("show");
  }
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.remove("show");
  }
}

// Global Delete Confirmation Modal Handler
function confirmDeleteExpense(id, description, amountFormatted) {
  const form = document.getElementById("deleteExpenseForm");
  const descEl = document.getElementById("deleteExpenseDesc");
  const amtEl = document.getElementById("deleteExpenseAmt");

  if (form && descEl && amtEl) {
    form.action = `/expenses/delete/${id}`;
    descEl.textContent = description;
    amtEl.textContent = amountFormatted;
    openModal("deleteModal");
  }
}

// Global View Transaction Details Modal
function viewExpenseDetails(expense) {
  const modal = document.getElementById("expenseDetailModal");
  if (!modal) return;

  document.getElementById("detailDesc").textContent = expense.description || "N/A";
  document.getElementById("detailAmount").textContent = expense.amount_formatted || `₹${expense.amount.toFixed(2)}`;
  document.getElementById("detailCategory").textContent = expense.category || "N/A";
  document.getElementById("detailDate").textContent = expense.date || "N/A";
  document.getElementById("detailPayment").textContent = expense.payment_method || "N/A";
  document.getElementById("detailNotes").textContent = expense.notes || "None";

  const anomalyStatusEl = document.getElementById("detailAnomalyStatus");
  const reasonsBox = document.getElementById("detailReasonsBox");
  const reasonsText = document.getElementById("detailReasonsText");
  const limitationText = document.getElementById("detailLimitationText");

  if (expense.is_anomaly) {
    anomalyStatusEl.innerHTML = `
      <span class="badge badge-anomaly">
        ⚠️ Potentially Unusual Spending
      </span>
      <span style="font-size: 0.8rem; color: #64748b; margin-left: 0.5rem;">
        Unusualness Score: <strong>${expense.anomaly_score !== null ? expense.anomaly_score : "N/A"}</strong>
      </span>
    `;
    reasonsBox.style.display = "block";
    const r = expense.reasons || {};
    reasonsText.textContent = `${r.summary || "Flagged by IsolationForest model."} ${r.category_comparison || ""}`;
    limitationText.textContent = r.limitation || "This reflects statistical divergence from history, not fraud.";
  } else {
    anomalyStatusEl.innerHTML = `
      <span class="badge badge-normal">
        ✓ Typical spending pattern
      </span>
    `;
    reasonsBox.style.display = "none";
  }

  // Edit Link
  const editBtn = document.getElementById("detailEditBtn");
  if (editBtn) {
    editBtn.href = `/expenses/edit/${expense.id}`;
  }

  openModal("expenseDetailModal");
}
