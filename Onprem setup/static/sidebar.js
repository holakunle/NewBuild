// sidebar.js
const coreMenuItemsConfig = [
    { id: 'menuDashboard', icon: 'fas fa-home', text: 'Dashboard', url: '/index.html' }
    // "Logout" removed from core items to ensure it's always last
];

async function fetchUserPermissions() {
    const token = localStorage.getItem('token') || sessionStorage.getItem('token');
    if (!token) {
        alert('No user logged in. Redirecting to login.');
        window.location.href = '/login.html';
        return [];
    }
    try {
        const response = await fetch('/api/current-user', {
            headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' }
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const userData = await response.json();
        return userData.permissions.map(perm => perm.name);
    } catch (error) {
        console.error('Error fetching user permissions:', error);
        alert('Failed to load user permissions. Please log in again.');
        window.location.href = '/login.html';
        return [];
    }
}

async function renderSidebar(pageSpecificItems = []) {
    const permissions = await fetchUserPermissions();
    const sidebarMenu = document.getElementById('sidebarMenu');
    sidebarMenu.innerHTML = '';

    // Combine core items (excluding Logout) and page-specific items
    const allMenuItems = [...coreMenuItemsConfig, ...pageSpecificItems].filter(item =>
        !item.permission || permissions.includes(item.permission)
    );

    // Render all items except Logout
    allMenuItems.forEach(item => {
        const menuItem = document.createElement('div');
        menuItem.className = `menu-item ${item.active ? 'active' : ''}`;
        menuItem.id = item.id;

        if (item.permission) menuItem.setAttribute('data-permission', item.permission);

        let innerHTML = `<i class="${item.icon}"></i> <span>${item.text}</span>`;
        if (item.badgeId) innerHTML += ` <span class="badge" id="${item.badgeId}">${item.badgeValue}</span>`;

        menuItem.innerHTML = innerHTML;

        if (item.url) {
            menuItem.addEventListener('click', () => window.location.href = item.url);
        }
        sidebarMenu.appendChild(menuItem);
    });

    // Always append Logout as the last item
    const logoutItem = document.createElement('div');
    logoutItem.className = 'menu-item';
    logoutItem.id = 'menuLogout';
    logoutItem.innerHTML = `<i class="fas fa-sign-out-alt"></i> <span>Logout</span>`;
    sidebarMenu.appendChild(logoutItem);

    attachSidebarEventListeners();
}

function attachSidebarEventListeners() {
    const menuDashboard = document.getElementById('menuDashboard');
    const menuLogout = document.getElementById('menuLogout');

    if (menuDashboard) menuDashboard.addEventListener('click', () => window.location.href = '/index.html');
    if (menuLogout) menuLogout.addEventListener('click', logout);
}

function logout() {
    localStorage.removeItem('token');
    sessionStorage.removeItem('token');
    window.location.href = '/login.html';
}

// Export functions for use in HTML files
export { renderSidebar, coreMenuItemsConfig, attachSidebarEventListeners };