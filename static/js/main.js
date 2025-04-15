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
    if (!taskIndicator) return; // Element not found
    
    if (isRunning) {
        taskIndicator.style.display = 'block';
        
        // If we're not on the home page, make it clickable
        if (window.location.pathname !== '/') {
            taskIndicator.classList.add('active');
        } else {
            taskIndicator.classList.remove('active');
        }
    } else {
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
