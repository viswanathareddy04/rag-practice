/**
 * Credit Card Insurance RAG — Frontend Application Logic
 * Consumes FastAPI SSE stream endpoints and renders dark fintech interface.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const cardSelect = document.getElementById("cardSelect");
    const indexStatus = document.getElementById("indexStatus");
    const modelBadge = document.getElementById("modelBadge");
    const suggestionChips = document.getElementById("suggestionChips");
    const chatScroll = document.getElementById("chatScroll");
    const chatMessages = document.getElementById("chatMessages");
    const queryForm = document.getElementById("queryForm");
    const queryInput = document.getElementById("queryInput");
    const submitBtn = document.getElementById("submitBtn");
    const activeCardIndicator = document.getElementById("activeCardIndicator");
    const sourcesCount = document.getElementById("sourcesCount");
    const citationsEmptyState = document.getElementById("citationsEmptyState");
    const sourcesList = document.getElementById("sourcesList");

    let cardsData = [];
    let selectedCard = null;
    let isStreaming = false;

    // Initialize application
    init();

    async function init() {
        await Promise.all([fetchStatus(), fetchCards()]);
        setupEventListeners();
    }

    async function fetchStatus() {
        try {
            const res = await fetch("/api/status");
            if (res.ok) {
                const data = await res.json();
                if (data.model && modelBadge) {
                    modelBadge.querySelector(".model-name").textContent = data.model;
                }
            }
        } catch (err) {
            console.warn("Status fetch failed:", err);
        }
    }

    async function fetchCards() {
        try {
            const res = await fetch("/api/cards");
            if (!res.ok) throw new Error("Failed to load cards list");
            cardsData = await res.json();

            cardSelect.innerHTML = "";
            if (cardsData.length === 0) {
                cardSelect.innerHTML = '<option value="" disabled>No insurance certificates found</option>';
                return;
            }

            cardsData.forEach((c) => {
                const opt = document.createElement("option");
                opt.value = c.card_id;
                const statusTag = c.is_indexed ? `[${c.total_chunks} chunks]` : "[unindexed]";
                opt.textContent = `${c.bank} — ${c.card_name} ${statusTag}`;
                cardSelect.appendChild(opt);
            });

            // Default select td_aeroplan_vip if present, else first card
            const preferred = cardsData.find((c) => c.card_id.includes("td_aeroplan") || c.card_id.includes("vip"));
            if (preferred) {
                cardSelect.value = preferred.card_id;
                selectedCard = preferred;
            } else {
                cardSelect.value = cardsData[0].card_id;
                selectedCard = cardsData[0];
            }

            updateCardDisplay();
        } catch (err) {
            console.error("Error fetching cards:", err);
            cardSelect.innerHTML = '<option value="" disabled>Error loading cards</option>';
        }
    }

    function updateCardDisplay() {
        if (!selectedCard) return;
        activeCardIndicator.textContent = `Active: ${selectedCard.bank} ${selectedCard.card_name}`;

        const statusDot = indexStatus.querySelector(".status-dot");
        const statusText = indexStatus.querySelector(".status-text");

        if (selectedCard.is_indexed) {
            indexStatus.className = "status-pill status-ready";
            statusText.textContent = `Indexed (${selectedCard.total_chunks || 0} chunks)`;
        } else {
            indexStatus.className = "status-pill status-indexing";
            statusText.textContent = "Unindexed (Auto on Query)";
        }
    }

    function setupEventListeners() {
        // Card change
        cardSelect.addEventListener("change", (e) => {
            selectedCard = cardsData.find((c) => c.card_id === e.target.value);
            updateCardDisplay();
        });

        // Quick prompt chips
        suggestionChips.addEventListener("click", (e) => {
            const chip = e.target.closest(".chip");
            if (chip && chip.dataset.query) {
                queryInput.value = chip.dataset.query;
                queryInput.focus();
                handleFormSubmit();
            }
        });

        // Auto-expand textarea & Enter to submit
        queryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleFormSubmit();
            }
        });

        // Query Form Submit
        queryForm.addEventListener("submit", (e) => {
            e.preventDefault();
            handleFormSubmit();
        });
    }

    function handleFormSubmit() {
        const query = queryInput.value.trim();
        if (!query || isStreaming || !selectedCard) return;
        executeStreamQuery(query);
    }

    async function executeStreamQuery(query) {
        isStreaming = true;
        submitBtn.disabled = true;
        queryInput.disabled = true;

        // Append User Message
        appendUserMessage(query);
        queryInput.value = "";

        // Append Assistant Message with Streaming Placeholder
        const assistantMsgEl = createAssistantMessageElement();
        chatMessages.appendChild(assistantMsgEl);
        scrollToBottom();

        const bodyEl = assistantMsgEl.querySelector(".message-body");
        const indicatorEl = assistantMsgEl.querySelector(".streaming-indicator");
        const timeEl = assistantMsgEl.querySelector(".message-time");
        const startTime = Date.now();

        let accumulatedText = "";

        try {
            const response = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    card_id: selectedCard.card_id,
                    query: query,
                }),
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop(); // Keep incomplete trailing chunk

                for (const block of lines) {
                    if (!block.trim()) continue;
                    const eventMatch = block.match(/^event:\s*(\w+)/m);
                    const dataMatch = block.match(/^data:\s*(.*)$/m);

                    if (eventMatch && dataMatch) {
                        const eventType = eventMatch[1];
                        const eventData = JSON.parse(dataMatch[1]);

                        if (eventType === "context") {
                            renderGroundingSources(eventData);
                        } else if (eventType === "token") {
                            accumulatedText += eventData;
                            bodyEl.innerHTML = renderMarkdown(accumulatedText);
                            scrollToBottom();
                        } else if (eventType === "citations") {
                            // Citations are already rendered in sources drawer
                        } else if (eventType === "done") {
                            const elapsed = eventData.elapsed_ms || Date.now() - startTime;
                            timeEl.textContent = `${(elapsed / 1000).toFixed(1)}s`;
                        }
                    }
                }
            }

            // If card was just indexed, mark it as indexed locally
            selectedCard.is_indexed = true;
            updateCardDisplay();

        } catch (err) {
            console.error("Stream error:", err);
            bodyEl.innerHTML = `<p style="color: #EF4444;"><strong>Error:</strong> Failed to retrieve policy information (${err.message}). Ensure backend server is reachable.</p>`;
        } finally {
            if (indicatorEl) indicatorEl.remove();
            isStreaming = false;
            submitBtn.disabled = false;
            queryInput.disabled = false;
            queryInput.focus();
            scrollToBottom();
        }
    }

    function appendUserMessage(text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message user-message";
        msgDiv.innerHTML = `
            <div class="message-avatar">
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-header">
                    <span class="sender-name">You</span>
                    <span class="message-time">${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                </div>
                <div class="message-body">
                    <p>${escapeHtml(text)}</p>
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
    }

    function createAssistantMessageElement() {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message assistant-message";
        msgDiv.innerHTML = `
            <div class="message-avatar">
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-header">
                    <span class="sender-name">Policy Assistant</span>
                    <span class="message-time">Streaming...</span>
                </div>
                <div class="message-body">
                    <div class="streaming-indicator">
                        <span class="pulse-dot"></span>
                        <span>Retrieving certificate clauses...</span>
                    </div>
                </div>
            </div>
        `;
        return msgDiv;
    }

    function renderGroundingSources(sources) {
        if (!sources || sources.length === 0) {
            citationsEmptyState.style.display = "flex";
            sourcesList.style.display = "none";
            sourcesCount.textContent = "0 Chunks";
            return;
        }

        citationsEmptyState.style.display = "none";
        sourcesList.style.display = "flex";
        sourcesList.innerHTML = "";
        sourcesCount.textContent = `${sources.length} Chunks`;

        sources.forEach((s) => {
            const card = document.createElement("div");
            card.className = "citation-card";
            card.innerHTML = `
                <div class="citation-card-header">
                    <span class="rank-badge">#${s.rank}</span>
                    <span class="page-badge">Page ${s.page_number}</span>
                    <span class="score-badge">Score: ${s.score}</span>
                </div>
                <div class="breadcrumb-trail">${escapeHtml(s.breadcrumb)}</div>
                <div class="chunk-snippet">${escapeHtml(s.snippet)}</div>
                <button type="button" class="expand-chunk-btn">View Full Section ▾</button>
                <div class="full-chunk-content">${escapeHtml(s.full_content || s.snippet)}</div>
            `;

            const expandBtn = card.querySelector(".expand-chunk-btn");
            const fullContent = card.querySelector(".full-chunk-content");
            expandBtn.addEventListener("click", () => {
                const isOpen = fullContent.style.display === "block";
                fullContent.style.display = isOpen ? "none" : "block";
                expandBtn.textContent = isOpen ? "View Full Section ▾" : "Collapse Section ▴";
            });

            sourcesList.appendChild(card);
        });
    }

    function renderMarkdown(raw) {
        if (!raw) return "";

        // Escape HTML tags to prevent XSS
        let html = raw
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Format Headers: ### Heading -> <h3>Heading</h3>
        html = html.replace(/^###\s+(.*$)/gim, "<h3>$1</h3>");

        // Format Citations tag: [Policy Name, Section X, Page Y]
        html = html.replace(/\[([^\]]+,\s*Page\s*\d+)\]/gi, '<span class="citation-tag">[$1]</span>');
        html = html.replace(/\[(Page\s*\d+[^\]]*)\]/gi, '<span class="citation-tag">[$1]</span>');

        // Format Bold: **text** -> <strong>text</strong>
        html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

        // Format Bullet list items: - item -> <li>item</li>
        html = html.replace(/^\s*-\s+(.*$)/gim, "<li>$1</li>");
        html = html.replace(/(<li>.*<\/li>)/gims, "<ul>$1</ul>");
        // Clean up nested <ul> tags from multiple regex matches
        html = html.replace(/<\/ul>\s*<ul>/g, "");

        // Convert double newlines to paragraph breaks
        const paragraphs = html.split(/\n\n+/);
        return paragraphs
            .map((p) => {
                p = p.trim();
                if (p.startsWith("<h3>") || p.startsWith("<ul>") || p.startsWith("<div")) {
                    return p;
                }
                return `<p>${p.replace(/\n/g, "<br>")}</p>`;
            })
            .join("");
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function scrollToBottom() {
        chatScroll.scrollTop = chatScroll.scrollHeight;
    }
});
