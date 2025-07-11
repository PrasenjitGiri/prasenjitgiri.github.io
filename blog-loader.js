// Blog posts configuration
const blogPosts = [
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
        <div class="timeline-item">
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

// Create year section HTML
function createYearSection(year, months) {
    const monthsHtml = Object.entries(months)
        .sort(([a], [b]) => new Date(`${a} 1, ${year}`) - new Date(`${b} 1, ${year}`))
        .reverse() // Most recent month first
        .map(([month, posts]) => {
            const postsHtml = posts
                .sort((a, b) => new Date(b.date) - new Date(a.date))
                .map(createTimelineItem)
                .join('');
            
            return `
                <div class="timeline-month">
                    <h3 class="month-header">${month} ${year}</h3>
                    <div class="timeline-items">
                        ${postsHtml}
                    </div>
                </div>
            `;
        })
        .join('');
    
    return `
        <div class="timeline-year">
            <h2 class="year-header">${year}</h2>
            ${monthsHtml}
        </div>
    `;
}

// Load all posts for blog page with timeline view
function loadAllPosts() {
    const blogPostsContainer = document.getElementById('blog-posts-container');
    if (blogPostsContainer) {
        // Sort posts by date (newest first)
        const sortedPosts = blogPosts.sort((a, b) => new Date(b.date) - new Date(a.date));
        
        // Group posts by year and month
        const groupedPosts = groupPostsByDate(sortedPosts);
        
        // Create timeline HTML
        const timelineHtml = Object.entries(groupedPosts)
            .sort(([a], [b]) => parseInt(b) - parseInt(a)) // Sort years descending
            .map(([year, months]) => createYearSection(year, months))
            .join('');
        
        blogPostsContainer.innerHTML = `
            <div class="blog-timeline">
                <div class="timeline-header">
                    <h2>All Posts</h2>
                    <p>A chronological journey through my insights and thoughts</p>
                </div>
                ${timelineHtml}
            </div>
        `;
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