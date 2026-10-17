const STORAGE_KEY = "todo-app-tasks";

const form = document.getElementById("todo-form");
const input = document.getElementById("todo-input");
const dateInput = document.getElementById("todo-date");
const datePicker = document.getElementById("todo-date-picker");
const list = document.getElementById("todo-list");
const emptyState = document.getElementById("empty-state");
const formError = document.getElementById("form-error");
const todoCount = document.getElementById("todo-count");

let tasks = loadTasks();

setDateFields(getTodayIso());
datePicker.addEventListener("change", handleDatePickerChange);
dateInput.addEventListener("input", handleDateTextInput);
form.addEventListener("submit", handleSubmit);
list.addEventListener("click", handleListClick);
list.addEventListener("change", handleListChange);

render();

function handleSubmit(event) {
  event.preventDefault();
  const text = sanitizeTaskText(input.value);
  const date = parseDisplayDate(dateInput.value);

  if (!text) {
    showFormError("Görev metni boş olamaz.");
    input.focus();
    return;
  }

  if (!date) {
    showFormError("Geçerli bir tarih seçin (GG/AA/YIL).");
    dateInput.focus();
    return;
  }

  formError.hidden = true;
  tasks.push({
    id: createId(),
    text: text,
    date: date,
    completed: false,
  });
  input.value = "";
  setDateFields(getTodayIso());
  persistAndRender();
  input.focus();
}

function handleListClick(event) {
  const deleteButton = event.target.closest(".delete-btn");
  if (!deleteButton) {
    return;
  }

  const id = deleteButton.closest(".todo-item").dataset.id;
  tasks = tasks.filter(function (task) {
    return task.id !== id;
  });
  persistAndRender();
}

function handleListChange(event) {
  if (!event.target.classList.contains("todo-checkbox")) {
    return;
  }

  const id = event.target.closest(".todo-item").dataset.id;
  const task = tasks.find(function (item) {
    return item.id === id;
  });

  if (!task) {
    return;
  }

  task.completed = event.target.checked;
  persistAndRender();
}

function handleDatePickerChange() {
  if (!isValidIsoDate(datePicker.value)) {
    return;
  }

  dateInput.value = formatDateDisplay(datePicker.value);
}

function handleDateTextInput() {
  const isoDate = parseDisplayDate(dateInput.value);
  datePicker.value = isoDate || "";
}

function persistAndRender() {
  saveTasks(tasks);
  render();
}

function render() {
  list.replaceChildren();

  tasks.forEach(function (task) {
    list.appendChild(createTaskItem(task));
  });

  emptyState.hidden = tasks.length > 0;
  todoCount.textContent = formatCount(tasks.length);
}

function createTaskItem(task) {
  const item = document.createElement("tr");
  item.className = "todo-item" + (task.completed ? " completed" : "");
  item.dataset.id = task.id;

  const taskCell = document.createElement("td");
  taskCell.className = "task-cell";

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.className = "todo-checkbox";
  checkbox.checked = task.completed;
  checkbox.setAttribute("aria-label", "Görevi tamamlandı olarak işaretle");

  const text = document.createElement("span");
  text.className = "todo-text";
  text.textContent = task.text;

  taskCell.append(checkbox, text);

  const dateCell = document.createElement("td");
  dateCell.className = "date-cell";
  dateCell.textContent = formatDateDisplay(task.date);

  const actionsCell = document.createElement("td");
  actionsCell.className = "actions-cell";

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.className = "delete-btn";
  deleteButton.textContent = "Sil";
  actionsCell.appendChild(deleteButton);

  item.append(taskCell, dateCell, actionsCell);
  return item;
}

function showFormError(message) {
  formError.textContent = message;
  formError.hidden = false;
}

function sanitizeTaskText(value) {
  return String(value).replace(/\s+/g, " ").trim();
}

function formatCount(count) {
  return count === 1 ? "1 görev" : count + " görev";
}

function createId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function setDateFields(isoDate) {
  dateInput.value = formatDateDisplay(isoDate);
  datePicker.value = isoDate;
}

function parseDisplayDate(value) {
  const match = String(value).trim().match(/^(\d{2})\/(\d{2})\/(\d{4})$/);
  if (!match) {
    return "";
  }

  const isoDate = match[3] + "-" + match[2] + "-" + match[1];
  return isValidIsoDate(isoDate) ? isoDate : "";
}

function getTodayIso() {
  const now = new Date();
  const year = String(now.getFullYear());
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return year + "-" + month + "-" + day;
}

function isValidIsoDate(value) {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    return false;
  }

  const parts = value.split("-");
  const year = Number(parts[0]);
  const month = Number(parts[1]);
  const day = Number(parts[2]);
  const date = new Date(year, month - 1, day);

  return (
    date.getFullYear() === year &&
    date.getMonth() === month - 1 &&
    date.getDate() === day
  );
}

function formatDateDisplay(isoDate) {
  if (!isValidIsoDate(isoDate)) {
    return "—";
  }

  const parts = isoDate.split("-");
  return parts[2] + "/" + parts[1] + "/" + parts[0];
}

function loadTasks() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }

    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed
      .filter(function (task) {
        return task && typeof task.text === "string";
      })
      .map(function (task) {
        return {
          id: typeof task.id === "string" ? task.id : createId(),
          text: sanitizeTaskText(task.text).slice(0, 200),
          date: isValidIsoDate(task.date) ? task.date : "",
          completed: Boolean(task.completed),
        };
      })
      .filter(function (task) {
        return task.text.length > 0;
      });
  } catch (error) {
    return [];
  }
}

function saveTasks(items) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
}
