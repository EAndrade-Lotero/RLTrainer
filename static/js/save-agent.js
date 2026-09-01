(function () {
  const EXPORT_URL = "/api/agent/export";
  const DEFAULT_SUBTITLE =
    "Choose whether to overwrite the existing agent or use a new file in saved_agents.";
  const APPLY_SUBTITLE =
    "Choose whether to overwrite the existing agent or use a new file in saved_agents before applying this configuration.";

  const dialog = document.getElementById("save-agent-dialog");
  const form = document.getElementById("save-agent-form");
  const filenameInput = document.getElementById("save-agent-filename");
  const overwriteHint = document.getElementById("save-agent-overwrite-hint");
  const subtitleEl = document.getElementById("save-agent-dialog-subtitle");
  const confirmButton = document.getElementById("save-agent-confirm");
  const saveButton = document.getElementById("save-agent-button");

  function statusElements() {
    return ["save-agent-status", "save-status"]
      .map((id) => document.getElementById(id))
      .filter(Boolean);
  }

  function setStatus(message, type) {
    statusElements().forEach((el) => {
      el.hidden = !message;
      el.textContent = message || "";
      el.classList.remove("setup-status--success", "setup-status--error");
      if (type) {
        el.classList.add(`setup-status--${type}`);
      }
    });
    if (window.ExperimentContext) {
      if (type === "error") {
        ExperimentContext.setStatus("error", "Error");
      } else if (type === "success") {
        ExperimentContext.setStatus("ready", "Ready");
      }
    }
  }

  function selectedSaveMode() {
    if (!form) {
      return "overwrite";
    }
    const checked = form.querySelector('input[name="save-agent-mode"]:checked');
    return checked ? checked.value : "overwrite";
  }

  function syncSaveFilenameField(preview) {
    if (!filenameInput) {
      return;
    }
    const mode = selectedSaveMode();
    if (mode === "new") {
      filenameInput.value = preview.suggested_filename || preview.default_filename || "";
      filenameInput.readOnly = false;
    } else {
      filenameInput.value = preview.current_filename || preview.default_filename || "";
      filenameInput.readOnly = true;
    }
  }

  function askSaveDestination(preview, options) {
    const settings = options || {};
    if (!dialog || !form || !filenameInput) {
      return Promise.resolve(null);
    }

    const currentName =
      preview.current_filename || preview.default_filename || "the current agent file";
    if (overwriteHint) {
      overwriteHint.textContent = preview.exists
        ? `This will replace ${currentName}.`
        : `No saved file yet. This will create ${currentName}.`;
    }
    if (subtitleEl) {
      subtitleEl.textContent = settings.subtitle || DEFAULT_SUBTITLE;
    }
    if (confirmButton) {
      confirmButton.textContent = settings.confirmLabel || "Save";
    }

    const overwriteRadio = form.querySelector(
      'input[name="save-agent-mode"][value="overwrite"]'
    );
    if (overwriteRadio) {
      overwriteRadio.checked = true;
    }
    syncSaveFilenameField(preview);
    dialog.returnValue = "";

    return new Promise((resolve) => {
      const onModeChange = () => syncSaveFilenameField(preview);
      form.querySelectorAll('input[name="save-agent-mode"]').forEach((input) => {
        input.addEventListener("change", onModeChange);
      });

      const cleanup = () => {
        form.querySelectorAll('input[name="save-agent-mode"]').forEach((input) => {
          input.removeEventListener("change", onModeChange);
        });
        form.removeEventListener("submit", onSubmit);
        dialog.removeEventListener("close", onClose);
      };

      const onSubmit = (event) => {
        const submitter = event.submitter;
        if (!submitter || submitter.value !== "confirm") {
          return;
        }
        const filename = filenameInput.value.trim();
        if (!filename) {
          event.preventDefault();
          setStatus(
            settings.emptyFilenameMessage || "Enter a file name before saving.",
            "error"
          );
        }
      };

      const onClose = () => {
        cleanup();
        if (dialog.returnValue !== "confirm") {
          resolve(null);
          return;
        }
        resolve({
          mode: selectedSaveMode(),
          filename: filenameInput.value.trim(),
        });
      };

      form.addEventListener("submit", onSubmit);
      dialog.addEventListener("close", onClose);
      dialog.showModal();
    });
  }

  async function fetchExportPreview() {
    const response = await fetch(EXPORT_URL, { cache: "no-store" });
    let data = {};
    try {
      data = await response.json();
    } catch (error) {
      data = {};
    }
    if (!response.ok) {
      throw new Error(data.error || "Could not inspect the current agent.");
    }
    return data;
  }

  async function promptSaveAgent(choice) {
    const response = await fetch(EXPORT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(choice || {}),
    });

    if (response.status === 404) {
      return { saved: false, skipped: true };
    }

    let data = {};
    try {
      data = await response.json();
    } catch (error) {
      data = {};
    }

    if (!response.ok) {
      throw new Error(data.error || "Could not save the agent.");
    }

    return {
      saved: true,
      skipped: false,
      path: data.path,
      filename: data.filename,
      kind: data.kind,
    };
  }

  async function saveCurrentAgent() {
    const preview = await fetchExportPreview();
    if (!preview.available) {
      throw new Error(
        preview.reason || "No agent is available to save for the current configuration."
      );
    }

    const choice = await askSaveDestination(preview, {
      confirmLabel: "Save",
      subtitle: DEFAULT_SUBTITLE,
    });
    if (!choice) {
      setStatus("Save cancelled.", null);
      return { saved: false, skipped: true, cancelled: true };
    }

    const result = await promptSaveAgent(choice);
    if (result.saved) {
      setStatus(`Agent saved to ${result.path}.`, "success");
    }
    return result;
  }

  if (saveButton) {
    saveButton.addEventListener("click", () => {
      saveCurrentAgent().catch((error) => {
        setStatus(error.message || "Could not save the agent.", "error");
      });
    });
  }

  window.SaveAgent = {
    fetchExportPreview,
    askSaveDestination,
    promptSaveAgent,
    saveCurrentAgent,
    setStatus,
    DEFAULT_SUBTITLE,
    APPLY_SUBTITLE,
  };
})();
