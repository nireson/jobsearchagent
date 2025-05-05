/**
 * Browser Agent UI - Main JavaScript
 * 
 * This file contains shared functionality used across the application.
 */

// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize socket.io connection
    initializeSocketConnection();
    
    // Check for running tasks on page load
    checkGlobalTaskStatus();
    
    // Set interval to periodically check for running tasks
    setInterval(checkGlobalTaskStatus, 10000); // Every 10 seconds

    // Fetch available models on page load
    fetchModels();
});

// Initialize Socket.IO connection
function initializeSocketConnection() {
    // This is initialized in the page-specific scripts
    if (typeof io !== 'undefined') {
        console.log('Socket.IO is available');
    } else {
        console.warn('Socket.IO is not loaded');
    }
}

// Format file size
function formatFileSize(bytes) {
    if (bytes < 1024) {
        return bytes + ' B';
    } else if (bytes < 1024 * 1024) {
        return (bytes / 1024).toFixed(1) + ' KB';
    } else {
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
}

// Format relative time (e.g., "2 minutes ago")
function formatRelativeTime(timestamp) {
    const now = new Date();
    const date = new Date(timestamp);
    const seconds = Math.floor((now - date) / 1000);
    
    let interval = Math.floor(seconds / 31536000);
    if (interval >= 1) {
        return interval + ' year' + (interval === 1 ? '' : 's') + ' ago';
    }
    
    interval = Math.floor(seconds / 2592000);
    if (interval >= 1) {
        return interval + ' month' + (interval === 1 ? '' : 's') + ' ago';
    }
    
    interval = Math.floor(seconds / 86400);
    if (interval >= 1) {
        return interval + ' day' + (interval === 1 ? '' : 's') + ' ago';
    }
    
    interval = Math.floor(seconds / 3600);
    if (interval >= 1) {
        return interval + ' hour' + (interval === 1 ? '' : 's') + ' ago';
    }
    
    interval = Math.floor(seconds / 60);
    if (interval >= 1) {
        return interval + ' minute' + (interval === 1 ? '' : 's') + ' ago';
    }
    
    return Math.floor(seconds) + ' second' + (seconds === 1 ? '' : 's') + ' ago';
}

// Show a notification
function showNotification(message, type = 'info') {
    // Create notification container if it doesn't exist
    let notificationContainer = document.getElementById('notification-container');
    if (!notificationContainer) {
        notificationContainer = document.createElement('div');
        notificationContainer.id = 'notification-container';
        notificationContainer.style.position = 'fixed';
        notificationContainer.style.top = '20px';
        notificationContainer.style.right = '20px';
        notificationContainer.style.zIndex = '1050';
        document.body.appendChild(notificationContainer);
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show`;
    notification.role = 'alert';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add notification to container
    notificationContainer.appendChild(notification);
    
    // Remove notification after 5 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notificationContainer.removeChild(notification);
        }, 150);
    }, 5000);
}

// Confirm dialog
function confirmDialog(message, callback) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('confirm-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'confirm-modal';
        modal.className = 'modal fade';
        modal.tabIndex = '-1';
        modal.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Confirm</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <p id="confirm-message"></p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" id="confirm-ok">OK</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }
    
    // Set message
    document.getElementById('confirm-message').textContent = message;
    
    // Set up callback
    const okButton = document.getElementById('confirm-ok');
    const modalInstance = new bootstrap.Modal(modal);
    
    const okClickHandler = () => {
        modalInstance.hide();
        callback(true);
        okButton.removeEventListener('click', okClickHandler);
    };
    
    okButton.addEventListener('click', okClickHandler);
    
    // Show modal
    modalInstance.show();
    
    // Handle cancel
    modal.addEventListener('hidden.bs.modal', function() {
        okButton.removeEventListener('click', okClickHandler);
    }, { once: true });
}

// Global task tracking
let globalTaskId = null;
let globalTaskRunning = false;

// Function to check if there are any running tasks
function checkGlobalTaskStatus() {
    fetch('/task_status')
        .then(response => response.json())
        .then(data => {
            if (data.has_active_tasks) {
                const taskIds = Object.keys(data.active_tasks);
                if (taskIds.length > 0) {
                    // Update global state
                    globalTaskId = taskIds[0];
                    globalTaskRunning = true;
                    
                    // Show task indicator in nav
                    updateTaskIndicator(true);
                } else {
                    globalTaskId = null;
                    globalTaskRunning = false;
                    updateTaskIndicator(false);
                }
            } else {
                globalTaskId = null;
                globalTaskRunning = false;
                updateTaskIndicator(false);
            }
        })
        .catch(error => {
            console.error('Error checking global task status:', error);
        });
}

// Update the navigation task indicator
function updateTaskIndicator(isRunning) {
    const taskIndicator = document.getElementById('taskIndicator');
    console.log(`[DEBUG main.js] updateTaskIndicator called with isRunning = ${isRunning}`); // Log call
    if (!taskIndicator) {
        console.log("[DEBUG main.js] Task indicator element not found.");
        return; // Element not found
    }
    
    if (isRunning) {
        console.log("[DEBUG main.js] Setting task indicator to display: block");
        taskIndicator.style.display = 'block';
        
        // If we're not on the home page, make it clickable
        if (window.location.pathname !== '/') {
            taskIndicator.classList.add('active');
        } else {
            taskIndicator.classList.remove('active');
        }
    } else {
        console.log("[DEBUG main.js] Setting task indicator to display: none");
        taskIndicator.style.display = 'none';
    }
}

// Expose functions for use in other scripts
window.taskManager = {
    getGlobalTaskId: function() {
        return globalTaskId;
    },
    isTaskRunning: function() {
        return globalTaskRunning;
    },
    setTaskRunning: function(taskId, isRunning) {
        globalTaskId = taskId;
        globalTaskRunning = isRunning;
        updateTaskIndicator(isRunning);
    },
    checkStatus: checkGlobalTaskStatus
};

// --- Utility Functions ---
function showToast(message, type = 'info') {
    // Basic toast implementation (replace with a library like Toastify if desired)
    console.log(`[${type.toUpperCase()}] ${message}`);
    // You could add a visual element here
}

// --- Prompt Management --- 
async function loadAndDisplayPrompts(query = '') {
    const listElement = document.getElementById('saved-prompts-list');
    if (!listElement) return;
    listElement.innerHTML = '<span class="text-muted small">Loading...</span>'; // Show loading indicator

    try {
        const url = query ? `/api/prompts/search?query=${encodeURIComponent(query)}` : '/api/prompts';
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const prompts = await response.json();
        displayPrompts(prompts);
    } catch (error) {
        console.error('Error loading prompts:', error);
        listElement.innerHTML = '<span class="text-danger small">Error loading prompts.</span>';
        showToast('Failed to load saved prompts', 'error');
    }
}

function displayPrompts(prompts) {
    const listElement = document.getElementById('saved-prompts-list');
    const promptInput = document.getElementById('prompt-input'); // Main prompt input
    if (!listElement || !promptInput) return;

    listElement.innerHTML = ''; // Clear current list

    if (prompts.length === 0) {
        listElement.innerHTML = '<span class="text-muted small">No prompts saved yet.</span>';
        return;
    }

    prompts.forEach(prompt => {
        const item = document.createElement('div');
        item.className = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center py-1 px-2';
        
        const textSpan = document.createElement('span');
        textSpan.textContent = prompt.text.length > 50 ? prompt.text.substring(0, 50) + '...' : prompt.text;
        textSpan.title = prompt.text; // Show full prompt on hover
        textSpan.style.cursor = 'default';
        textSpan.style.flexGrow = '1';
        textSpan.style.marginRight = '10px';
        textSpan.style.overflow = 'hidden';
        textSpan.style.textOverflow = 'ellipsis';
        textSpan.style.whiteSpace = 'nowrap';

        const buttonGroup = document.createElement('div');
        buttonGroup.className = 'btn-group btn-group-sm';

        const loadButton = document.createElement('button');
        loadButton.className = 'btn btn-outline-primary btn-sm';
        loadButton.innerHTML = '<i class="fas fa-arrow-up"></i> Load'; // Changed text
        loadButton.title = 'Load this prompt into the search box';
        loadButton.onclick = () => {
            promptInput.value = prompt.text;
            showToast('Prompt loaded into search box');
            promptInput.focus(); // Focus the input after loading
        };

        const deleteButton = document.createElement('button');
        deleteButton.className = 'btn btn-outline-danger btn-sm';
        deleteButton.innerHTML = '<i class="fas fa-trash-alt"></i>';
        deleteButton.title = 'Delete this prompt';
        deleteButton.onclick = async () => {
            if (confirm('Are you sure you want to delete this prompt?')) {
                try {
                    const response = await fetch(`/api/prompts/${prompt.id}`, { method: 'DELETE' });
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    showToast('Prompt deleted');
                    loadAndDisplayPrompts(document.getElementById('search-prompts-input').value); // Refresh list with current search
                } catch (error) {
                    console.error('Error deleting prompt:', error);
                    showToast('Failed to delete prompt', 'error');
                }
            }
        };

        buttonGroup.appendChild(loadButton);
        buttonGroup.appendChild(deleteButton);

        item.appendChild(textSpan);
        item.appendChild(buttonGroup);
        listElement.appendChild(item);
    });
}

// --- End Agent Toggle Logic ---

// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    // Check if we are on the home page before adding home-specific listeners
    if (window.location.pathname === '/') {
        console.log("Initializing localStorage check for Home page...");

        // Load prompt from localStorage if present (from prompts page)
        const promptToLoad = localStorage.getItem('promptToLoad');
        const promptInput = document.getElementById('promptInput'); // Main prompt input on Home page
        if (promptToLoad && promptInput) {
            console.log("Found prompt to load from localStorage:", promptToLoad);
            promptInput.value = promptToLoad;
            localStorage.removeItem('promptToLoad'); // Clear after loading
            showToast('Prompt loaded from Saved Prompts page');
        }

        // Add other home-specific listeners here ONLY if they are not handled elsewhere (e.g., in index.html)

    }
    // NOTE: The main task form submission listener is now handled within templates/index.html
});
