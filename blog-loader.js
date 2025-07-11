// Blog posts configuration
const blogPosts = [
    {
        id: 'residual-analysis-timeseries',
        title: 'Residual Analysis in Time Series: Validating Models and Uncovering Hidden Patterns',
        date: '2025-01-25',
        excerpt: 'Deep dive into residual analysis for time series models, covering statistical and machine learning approaches to validate, improve, and build confidence in forecasting systems.',
        file: 'posts/residual-analysis-timeseries.md',
        tags: ['Time Series', 'Residual Analysis', 'Model Validation', 'Statistics', 'Machine Learning', 'Diagnostics']
    },
    {
        id: 'cyclical-encoding-deep-dive',
        title: 'Cyclical Encoding: Why Your Machine Learning Model Thinks December and January Are Worlds Apart',
        date: '2021-03-15',
        excerpt: 'A creative exploration of cyclical encoding using sine and cosine transformations. Discover why traditional integer encoding fails for time-based features and how trigonometry saves the day.',
        file: 'posts/cyclical-encoding-deep-dive.md',
        tags: ['Feature Engineering', 'Cyclical Encoding', 'Time Series', 'Machine Learning', 'Data Preprocessing', 'Trigonometry']
    },
    {
        id: 'timeseries-comprehensive-guide',
        title: 'Time Series Analysis: From Statistical Foundations to Neural Networks',
        date: '2020-01-20',
        excerpt: 'A comprehensive guide covering the entire spectrum of time series analysis from classical statistical methods to cutting-edge neural network architectures.',
        file: 'posts/timeseries-comprehensive-guide.md',
        tags: ['Time Series', 'Statistics', 'Machine Learning', 'Neural Networks', 'ARIMA', 'LSTM']
    },
    {
        id: 'agentic-ai-future',
        title: 'The Future of Agentic AI: Building Autonomous Intelligence',
        date: '2025-01-15',
        excerpt: 'Exploring the evolution of agentic AI systems and their potential to revolutionize how we interact with artificial intelligence.',
        file: 'posts/agentic-ai-future.md',
        tags: ['Agentic AI', 'Autonomous Systems', 'AI Strategy']
    },
    {
        id: 'vector-indexing-strategies',
        title: 'Vector Indexing Strategies: Choosing the Right Approach for Your Use Case',
        date: '2025-01-10',
        excerpt: 'A comprehensive guide to vector database indexing strategies, performance metrics, and when to use different approaches.',
        file: 'posts/vector-indexing-strategies.md',
        tags: ['Vector Databases', 'Performance', 'Search']
    },
    {
        id: 'llm-output-metrics',
        title: 'Evaluating LLM Outputs: Metrics That Matter',
        date: '2025-01-05',
        excerpt: 'My perspective on measuring LLM performance beyond traditional metrics, focusing on practical evaluation frameworks.',
        file: 'posts/llm-output-metrics.md',
        tags: ['LLM Evaluation', 'Metrics', 'Quality Assessment']
    },
    {
        id: 'llm-observability',
        title: 'LLM Observability: Monitoring AI in Production',
        date: '2024-12-28',
        excerpt: 'Building comprehensive observability systems for LLM applications to ensure reliability and performance in production.',
        file: 'posts/llm-observability.md',
        tags: ['LLM Observability', 'Monitoring', 'Production AI']
    }
];

// Global variables for search and filter state
let allPosts = [];
let filteredPosts = [];
let expandedSections = new Set();

// Format date for display
function formatDate(dateString) {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateString).toLocaleDateString('en-US', options);
}

// Create blog card HTML
function createBlogCard(post) {
    return `
        <article class="blog-card">
            <div class="blog-card-content">
                <h3><a href="post.html?id=${post.id}" class="read-more">${post.title}</a></h3>
                <p class="excerpt">${post.excerpt}</p>
                <div class="blog-meta">
                    <span class="date">${formatDate(post.date)}</span>
                    <a href="post.html?id=${post.id}" class="read-more">Read More →</a>
                </div>
            </div>
        </article>
    `;
}

// Create blog item HTML for blog page
function createBlogItem(post) {
    return `
        <article class="blog-item">
            <h3><a href="post.html?id=${post.id}" class="read-more">${post.title}</a></h3>
            <p class="excerpt">${post.excerpt}</p>
            <div class="blog-meta">
                <span class="date">${formatDate(post.date)}</span>
                <span class="tags">${post.tags.join(', ')}</span>
                <a href="post.html?id=${post.id}" class="read-more">Read Full Post →</a>
            </div>
        </article>
    `;
}

// Load latest posts for homepage
function loadLatestPosts() {
    const latestPostsContainer = document.getElementById('latest-posts');
    if (latestPostsContainer) {
        // Show latest 3 posts
        const latestPosts = blogPosts.slice(0, 3);
        latestPostsContainer.innerHTML = latestPosts.map(createBlogCard).join('');
    }
}

// Group posts by year and month
function groupPostsByDate(posts) {
    const grouped = {};
    
    posts.forEach(post => {
        const date = new Date(post.date);
        const year = date.getFullYear();
        const month = date.toLocaleDateString('en-US', { month: 'long' });
        
        if (!grouped[year]) {
            grouped[year] = {};
        }
        
        if (!grouped[year][month]) {
            grouped[year][month] = [];
        }
        
        grouped[year][month].push(post);
    });
    
    return grouped;
}

// Create timeline item HTML
function createTimelineItem(post) {
    return `
        <div class="timeline-item" data-post-id="${post.id}">
            <div class="timeline-date">
                <span class="day">${new Date(post.date).getDate()}</span>
            </div>
            <div class="timeline-content">
                <h3><a href="post.html?id=${post.id}" class="read-more">${post.title}</a></h3>
                <p class="excerpt">${post.excerpt}</p>
                <div class="timeline-meta">
                    <span class="tags">${post.tags.join(', ')}</span>
                    <a href="post.html?id=${post.id}" class="read-more">Read Full Post →</a>
                </div>
            </div>
        </div>
    `;
}

// Get the most recent year and month
function getMostRecentYearMonth(posts) {
    if (posts.length === 0) return { year: null, month: null };
    
    const sortedPosts = posts.sort((a, b) => new Date(b.date) - new Date(a.date));
    const mostRecent = sortedPosts[0];
    const date = new Date(mostRecent.date);
    
    return {
        year: date.getFullYear(),
        month: date.toLocaleDateString('en-US', { month: 'long' })
    };
}

// Create month section HTML with collapsible functionality
function createMonthSection(month, year, posts, isExpanded = false) {
    const monthId = `month-${year}-${month.replace(/\s+/g, '-')}`;
    const postsHtml = posts
        .sort((a, b) => new Date(b.date) - new Date(a.date))
        .map(createTimelineItem)
        .join('');
    
    return `
        <div class="timeline-month" data-month-id="${monthId}">
            <h3 class="month-header ${isExpanded ? 'expanded' : 'collapsed'}" onclick="toggleMonth('${monthId}')">
                <span class="expand-icon">${isExpanded ? '▼' : '▶'}</span>
                ${month} ${year}
                <span class="post-count">(${posts.length} post${posts.length !== 1 ? 's' : ''})</span>
            </h3>
            <div class="timeline-items ${isExpanded ? 'expanded' : 'collapsed'}" id="${monthId}">
                ${postsHtml}
            </div>
        </div>
    `;
}

// Create year section HTML with collapsible functionality
function createYearSection(year, months, mostRecentYear, mostRecentMonth) {
    const yearId = `year-${year}`;
    const isCurrentYear = year == mostRecentYear;
    
    const monthsHtml = Object.entries(months)
        .sort(([a], [b]) => new Date(`${a} 1, ${year}`) - new Date(`${b} 1, ${year}`))
        .reverse() // Most recent month first
        .map(([month, posts]) => {
            const isCurrentMonth = isCurrentYear && month === mostRecentMonth;
            return createMonthSection(month, year, posts, isCurrentMonth);
        })
        .join('');
    
    return `
        <div class="timeline-year" data-year-id="${yearId}">
            <h2 class="year-header ${isCurrentYear ? 'expanded' : 'collapsed'}" onclick="toggleYear('${yearId}')">
                <span class="expand-icon">${isCurrentYear ? '▼' : '▶'}</span>
                ${year}
                <span class="year-stats">(${Object.values(months).flat().length} posts)</span>
            </h2>
            <div class="year-content ${isCurrentYear ? 'expanded' : 'collapsed'}" id="${yearId}">
                ${monthsHtml}
            </div>
        </div>
    `;
}

// Search functionality
function searchPosts(query) {
    if (!query.trim()) {
        filteredPosts = [...allPosts];
    } else {
        const searchTerm = query.toLowerCase();
        filteredPosts = allPosts.filter(post => 
            post.title.toLowerCase().includes(searchTerm) ||
            post.excerpt.toLowerCase().includes(searchTerm) ||
            post.tags.some(tag => tag.toLowerCase().includes(searchTerm))
        );
    }
    renderFilteredTimeline();
}

// Render filtered timeline
function renderFilteredTimeline() {
    const blogPostsContainer = document.getElementById('blog-posts-container');
    if (!blogPostsContainer) return;
    
    if (filteredPosts.length === 0) {
        blogPostsContainer.innerHTML = `
            <div class="blog-timeline">
                <div class="timeline-header">
                    <h2>No Posts Found</h2>
                    <p>No posts match your search criteria. Try different keywords or clear the search.</p>
                </div>
            </div>
        `;
        return;
    }
    
    // Sort posts by date (newest first)
    const sortedPosts = filteredPosts.sort((a, b) => new Date(b.date) - new Date(a.date));
    
    // Group posts by year and month
    const groupedPosts = groupPostsByDate(sortedPosts);
    
    // Get most recent year and month
    const { year: mostRecentYear, month: mostRecentMonth } = getMostRecentYearMonth(sortedPosts);
    
    // Create timeline HTML
    const timelineHtml = Object.entries(groupedPosts)
        .sort(([a], [b]) => parseInt(b) - parseInt(a)) // Sort years descending
        .map(([year, months]) => createYearSection(year, months, mostRecentYear, mostRecentMonth))
        .join('');
    
    const resultCount = filteredPosts.length;
    const totalCount = allPosts.length;
    const showingText = resultCount === totalCount ? 
        `Showing all ${totalCount} posts` : 
        `Showing ${resultCount} of ${totalCount} posts`;
    
    blogPostsContainer.innerHTML = `
        <div class="blog-timeline">
            <div class="timeline-header">
                <h2>Blog Timeline</h2>
                <p>A chronological journey through my insights and thoughts</p>
                <div class="search-results-info">${showingText}</div>
            </div>
            ${timelineHtml}
        </div>
    `;
}

// Toggle functions
function toggleMonth(monthId) {
    const monthElement = document.getElementById(monthId);
    const headerElement = document.querySelector(`[data-month-id="${monthId}"] .month-header`);
    const iconElement = headerElement.querySelector('.expand-icon');
    
    if (monthElement.classList.contains('expanded')) {
        monthElement.classList.remove('expanded');
        monthElement.classList.add('collapsed');
        headerElement.classList.remove('expanded');
        headerElement.classList.add('collapsed');
        iconElement.textContent = '▶';
        expandedSections.delete(monthId);
    } else {
        monthElement.classList.remove('collapsed');
        monthElement.classList.add('expanded');
        headerElement.classList.remove('collapsed');
        headerElement.classList.add('expanded');
        iconElement.textContent = '▼';
        expandedSections.add(monthId);
    }
}

function toggleYear(yearId) {
    const yearElement = document.getElementById(yearId);
    const headerElement = document.querySelector(`[data-year-id="${yearId}"] .year-header`);
    const iconElement = headerElement.querySelector('.expand-icon');
    
    if (yearElement.classList.contains('expanded')) {
        yearElement.classList.remove('expanded');
        yearElement.classList.add('collapsed');
        headerElement.classList.remove('expanded');
        headerElement.classList.add('collapsed');
        iconElement.textContent = '▶';
        expandedSections.delete(yearId);
    } else {
        yearElement.classList.remove('collapsed');
        yearElement.classList.add('expanded');
        headerElement.classList.remove('collapsed');
        headerElement.classList.add('expanded');
        iconElement.textContent = '▼';
        expandedSections.add(yearId);
    }
}

function expandAll() {
    document.querySelectorAll('.year-content, .timeline-items').forEach(element => {
        element.classList.remove('collapsed');
        element.classList.add('expanded');
    });
    
    document.querySelectorAll('.year-header, .month-header').forEach(header => {
        header.classList.remove('collapsed');
        header.classList.add('expanded');
        const icon = header.querySelector('.expand-icon');
        if (icon) icon.textContent = '▼';
    });
    
    // Update expanded sections set
    expandedSections.clear();
    document.querySelectorAll('.year-content, .timeline-items').forEach(element => {
        expandedSections.add(element.id);
    });
}

function collapseAll() {
    document.querySelectorAll('.year-content, .timeline-items').forEach(element => {
        element.classList.remove('expanded');
        element.classList.add('collapsed');
    });
    
    document.querySelectorAll('.year-header, .month-header').forEach(header => {
        header.classList.remove('expanded');
        header.classList.add('collapsed');
        const icon = header.querySelector('.expand-icon');
        if (icon) icon.textContent = '▶';
    });
    
    expandedSections.clear();
}

// Load all posts for blog page with enhanced timeline view
function loadAllPosts() {
    const blogPostsContainer = document.getElementById('blog-posts-container');
    if (blogPostsContainer) {
        allPosts = [...blogPosts];
        filteredPosts = [...allPosts];
        
        // Create search and controls HTML
        const searchControlsHtml = `
            <div class="timeline-controls">
                <div class="search-container">
                    <input type="text" id="blog-search" placeholder="Search posts by title, content, or tags..." />
                    <button id="clear-search" onclick="clearSearch()">Clear</button>
                </div>
                <div class="expand-controls">
                    <button onclick="expandAll()">Expand All</button>
                    <button onclick="collapseAll()">Collapse All</button>
                </div>
            </div>
        `;
        
        // Insert controls before rendering timeline
        blogPostsContainer.innerHTML = searchControlsHtml;
        
        // Add search event listener
        setTimeout(() => {
            const searchInput = document.getElementById('blog-search');
            if (searchInput) {
                searchInput.addEventListener('input', (e) => searchPosts(e.target.value));
                searchInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        searchPosts(e.target.value);
                    }
                });
            }
        }, 100);
        
        // Render initial timeline
        renderFilteredTimeline();
    }
}

function clearSearch() {
    const searchInput = document.getElementById('blog-search');
    if (searchInput) {
        searchInput.value = '';
        searchPosts('');
    }
}

// Load individual post
async function loadPost() {
    const urlParams = new URLSearchParams(window.location.search);
    const postId = urlParams.get('id');
    
    if (!postId) {
        window.location.href = 'blog.html';
        return;
    }
    
    const post = blogPosts.find(p => p.id === postId);
    if (!post) {
        window.location.href = 'blog.html';
        return;
    }
    
    // Update page title
    document.title = `${post.title} - Prasenjit Giri`;
    
    // Update article header
    const articleTitle = document.getElementById('article-title');
    const articleMeta = document.getElementById('article-meta');
    
    if (articleTitle) {
        articleTitle.textContent = post.title;
    }
    
    if (articleMeta) {
        articleMeta.innerHTML = `
            <span>Published on ${formatDate(post.date)}</span>
            <span class="tags">Tags: ${post.tags.join(', ')}</span>
        `;
    }
    
    // Load and render markdown content
    try {
        const response = await fetch(post.file);
        if (!response.ok) {
            throw new Error('Failed to load post content');
        }
        
        const markdownContent = await response.text();
        const htmlContent = marked.parse(markdownContent);
        
        const articleContent = document.getElementById('article-content');
        if (articleContent) {
            articleContent.innerHTML = htmlContent;
        }
    } catch (error) {
        console.error('Error loading post:', error);
        const articleContent = document.getElementById('article-content');
        if (articleContent) {
            articleContent.innerHTML = `
                <p>Sorry, there was an error loading this post. Please try again later.</p>
                <p><a href="blog.html">← Back to Blog</a></p>
            `;
        }
    }
}

// Make functions globally available
window.toggleMonth = toggleMonth;
window.toggleYear = toggleYear;
window.expandAll = expandAll;
window.collapseAll = collapseAll;
window.clearSearch = clearSearch;

// Initialize based on current page
document.addEventListener('DOMContentLoaded', function() {
    const currentPage = window.location.pathname.split('/').pop();
    
    switch (currentPage) {
        case 'index.html':
        case '':
            loadLatestPosts();
            break;
        case 'blog.html':
            loadAllPosts();
            break;
        case 'post.html':
            loadPost();
            break;
    }
}); 