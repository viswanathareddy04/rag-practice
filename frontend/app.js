/**
 * Credit Card Insurance RAG — Frontend Application Logic
 * Supports Single-Card Policy Assistant & Multi-Card Side-by-Side Comparison Matrix.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Navigation & Modes
    const singleCardModeBtn = document.getElementById("singleCardModeBtn");
    const compareModeBtn = document.getElementById("compareModeBtn");
    const singleCardGroup = document.getElementById("singleCardGroup");
    const multiCardGroup = document.getElementById("multiCardGroup");
    const cardPillsSelector = document.getElementById("cardPillsSelector");
    const selectedCardCount = document.getElementById("selectedCardCount");
    const selectionHint = document.getElementById("selectionHint");

    // DOM Elements - Status & Workspace
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

    // Application State
    let currentMode = "single"; // "single" | "compare"
    let cardsData = [];
    let selectedCard = null;
    let selectedCompareCards = new Set();
    let isStreaming = false;

    // Single vs Comparison Suggestion Prompts
    const SINGLE_PROMPTS = [
        { label: "Flight Delay Limit", query: "What is the minimum delay time and maximum coverage for flight delay?" },
        { label: "Pre-Existing Stability", query: "What are the pre-existing medical condition stability requirements?" },
        { label: "Rental Car MSRP Cap", query: "What is the maximum MSRP for rental car collision damage coverage?" },
        { label: "Mobile Device Limit", query: "What is the mobile device insurance maximum limit and deductible?" },
        { label: "Pet Medical Cover?", query: "Does this insurance certificate cover pet health emergencies during travel?" }
    ];

    const COMPARE_PROMPTS = [
        { label: "Rental Car MSRP & Limits", query: "Compare maximum rental car collision coverage limits, vehicle MSRP caps, and rental durations." },
        { label: "Travel Medical Days (65+)", query: "Compare out-of-province emergency medical coverage durations and terms for travellers age 65 and older." },
        { label: "Trip Cancellation Payouts", query: "Compare trip cancellation maximum reimbursement limits per traveler and per trip." },
        { label: "Baggage Delay Thresholds", query: "Compare baggage delay minimum hours threshold and maximum essential purchases allowance." },
        { label: "Mobile Device Deductibles", query: "Compare mobile device insurance maximum coverage limit, replacement terms, and deductibles." }
    ];

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

            // Default select top 2 cards for comparison mode
            selectedCompareCards.clear();
            const defaultCompare = cardsData.slice(0, 3);
            defaultCompare.forEach(c => selectedCompareCards.add(c.card_id));

            renderCardPills();
            updateCardDisplay();
        } catch (err) {
            console.error("Error fetching cards:", err);
            cardSelect.innerHTML = '<option value="" disabled>Error loading cards</option>';
        }
    }

    function renderCardPills() {
        if (!cardPillsSelector) return;
        cardPillsSelector.innerHTML = "";

        cardsData.forEach((c) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = `card-pill-btn ${selectedCompareCards.has(c.card_id) ? "selected" : ""}`;
            btn.dataset.cardId = c.card_id;

            const check = document.createElement("span");
            check.className = "card-pill-check";
            check.textContent = selectedCompareCards.has(c.card_id) ? "✓" : "+";

            const name = document.createElement("span");
            name.textContent = `${c.bank} ${c.card_name.replace("Visa Infinite Privilege Card", "VIP").replace("Card", "")}`;

            btn.appendChild(check);
            btn.appendChild(name);

            btn.addEventListener("click", () => toggleCardSelection(c.card_id));
            cardPillsSelector.appendChild(btn);
        });

        updatePillHeader();
    }

    function toggleCardSelection(cardId) {
        if (selectedCompareCards.has(cardId)) {
            if (selectedCompareCards.size <= 2) {
                flashHint("Minimum 2 cards required for comparison");
                return;
            }
            selectedCompareCards.delete(cardId);
        } else {
            if (selectedCompareCards.size >= 3) {
                flashHint("Maximum 3 cards can be compared simultaneously");
                return;
            }
            selectedCompareCards.add(cardId);
        }

        renderCardPills();
        updateCardDisplay();
    }

    function updatePillHeader() {
        if (selectedCardCount) {
            selectedCardCount.textContent = selectedCompareCards.size;
        }
        if (selectionHint) {
            if (selectedCompareCards.size < 2) {
                selectionHint.textContent = "Select at least 2 cards";
                selectionHint.className = "selection-hint warning";
            } else {
                selectionHint.textContent = `${selectedCompareCards.size} cards active`;
                selectionHint.className = "selection-hint";
            }
        }
    }

    function flashHint(msg) {
        if (!selectionHint) return;
        const originalText = selectionHint.textContent;
        selectionHint.textContent = msg;
        selectionHint.className = "selection-hint warning";
        setTimeout(() => {
            updatePillHeader();
        }, 2500);
    }

    function updateCardDisplay() {
        if (currentMode === "single") {
            if (!selectedCard) return;
            activeCardIndicator.textContent = `Active: ${selectedCard.bank} ${selectedCard.card_name}`;

            if (selectedCard.is_indexed) {
                indexStatus.className = "status-pill status-ready";
                indexStatus.querySelector(".status-text").textContent = `Indexed (${selectedCard.total_chunks || 0} chunks)`;
            } else {
                indexStatus.className = "status-pill status-indexing";
                indexStatus.querySelector(".status-text").textContent = "Unindexed (Auto on Query)";
            }
        } else {
            const cardNames = Array.from(selectedCompareCards).map(id => {
                const c = cardsData.find(item => item.card_id === id);
                return c ? `${c.bank} VIP` : id;
            });
            activeCardIndicator.textContent = `Comparing: ${cardNames.join(" vs ")}`;
            indexStatus.className = "status-pill status-ready";
            indexStatus.querySelector(".status-text").textContent = `Multi-Card Matrix (${selectedCompareCards.size} cards)`;
        }
    }

    function renderSuggestionChips(prompts) {
        if (!suggestionChips) return;
        suggestionChips.innerHTML = "";
        prompts.forEach((item) => {
            const btn = document.createElement("button");
            btn.className = "chip";
            btn.dataset.query = item.query;
            btn.textContent = item.label;
            suggestionChips.appendChild(btn);
        });
    }

    function setMode(mode) {
        currentMode = mode;
        if (mode === "single") {
            singleCardModeBtn.classList.add("active");
            singleCardModeBtn.setAttribute("aria-selected", "true");
            compareModeBtn.classList.remove("active");
            compareModeBtn.setAttribute("aria-selected", "false");

            singleCardGroup.style.display = "block";
            multiCardGroup.style.display = "none";

            queryInput.placeholder = "Ask any question about coverage, limits, exclusions, or deductibles...";
            renderSuggestionChips(SINGLE_PROMPTS);
        } else {
            compareModeBtn.classList.add("active");
            compareModeBtn.setAttribute("aria-selected", "true");
            singleCardModeBtn.classList.remove("active");
            singleCardModeBtn.setAttribute("aria-selected", "false");

            singleCardGroup.style.display = "none";
            multiCardGroup.style.display = "block";

            queryInput.placeholder = "Compare specific coverage riders across selected cards (e.g. rental car collision, flight delay)...";
            renderSuggestionChips(COMPARE_PROMPTS);
            renderCardPills();
        }
        updateCardDisplay();
    }

    function setupEventListeners() {
        // Mode switching
        singleCardModeBtn.addEventListener("click", () => setMode("single"));
        compareModeBtn.addEventListener("click", () => setMode("compare"));

        // Single Card change
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
                queryInput.selectionStart = queryInput.selectionEnd = queryInput.value.length;
                queryInput.style.height = "auto";
                queryInput.style.height = `${Math.min(queryInput.scrollHeight, 120)}px`;
            }
        });

        // Auto-expand textarea & Enter to submit
        queryInput.addEventListener("input", () => {
            queryInput.style.height = "auto";
            queryInput.style.height = `${Math.min(queryInput.scrollHeight, 120)}px`;
        });

        queryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleFormSubmit();
            }
        });

        // Form Submit
        queryForm.addEventListener("submit", (e) => {
            e.preventDefault();
            handleFormSubmit();
        });
    }

    function handleFormSubmit() {
        const query = queryInput.value.trim();
        if (!query || isStreaming) return;

        if (currentMode === "single") {
            if (!selectedCard) return;
            executeStreamQuery(query);
        } else {
            if (selectedCompareCards.size < 2) {
                flashHint("Please select at least 2 cards to compare");
                return;
            }
            executeCompareStreamQuery(query);
        }
    }

    async function executeStreamQuery(query) {
        isStreaming = true;
        submitBtn.disabled = true;
        queryInput.disabled = true;

        appendUserMessage(query);
        queryInput.value = "";
        queryInput.style.height = "auto";

        const assistantMsgEl = createAssistantMessageElement(false);
        chatMessages.appendChild(assistantMsgEl);
        scrollToBottom();

        const bodyEl = assistantMsgEl.querySelector(".message-body");
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

            if (!response.ok) throw new Error(`Server returned HTTP ${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop() || "";

                for (const chunk of lines) {
                    if (!chunk.trim()) continue;
                    const eventMatch = chunk.match(/event:\s*(\w+)/);
                    const dataMatch = chunk.match(/data:\s*(.*)/s);
                    if (!eventMatch || !dataMatch) continue;

                    const event = eventMatch[1];
                    const data = JSON.parse(dataMatch[1]);

                    if (event === "context") {
                        renderGroundingSources(data);
                    } else if (event === "token") {
                        accumulatedText += data;
                        bodyEl.innerHTML = renderMarkdown(accumulatedText);
                        scrollToBottom();
                    } else if (event === "citations") {
                        // Citations final pass
                    } else if (event === "done") {
                        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                        timeEl.textContent = `${elapsed}s`;
                    }
                }
            }
        } catch (err) {
            console.error("Stream error:", err);
            bodyEl.innerHTML = `<p style="color: #EF4444;">Error generating response: ${escapeHtml(err.message)}</p>`;
        } finally {
            isStreaming = false;
            submitBtn.disabled = false;
            queryInput.disabled = false;
            queryInput.focus();
        }
    }

    async function executeCompareStreamQuery(query) {
        isStreaming = true;
        submitBtn.disabled = true;
        queryInput.disabled = true;

        const activeCardIds = Array.from(selectedCompareCards);
        const cardDisplayNames = activeCardIds.map(id => {
            const c = cardsData.find(item => item.card_id === id);
            return c ? `${c.bank} ${c.card_name.replace("Visa Infinite Privilege Card", "VIP")}` : id;
        });

        appendUserMessage(query, cardDisplayNames);
        queryInput.value = "";
        queryInput.style.height = "auto";

        const assistantMsgEl = createAssistantMessageElement(true);
        chatMessages.appendChild(assistantMsgEl);
        scrollToBottom();

        const bodyEl = assistantMsgEl.querySelector(".message-body");
        const timeEl = assistantMsgEl.querySelector(".message-time");
        const startTime = Date.now();
        let accumulatedText = "";

        try {
            const response = await fetch("/api/compare", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    card_ids: activeCardIds,
                    query: query,
                }),
            });

            if (!response.ok) throw new Error(`Server returned HTTP ${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop() || "";

                for (const chunk of lines) {
                    if (!chunk.trim()) continue;
                    const eventMatch = chunk.match(/event:\s*(\w+)/);
                    const dataMatch = chunk.match(/data:\s*(.*)/s);
                    if (!eventMatch || !dataMatch) continue;

                    const event = eventMatch[1];
                    const data = JSON.parse(dataMatch[1]);

                    if (event === "cards_context") {
                        renderSegmentedSources(data);
                    } else if (event === "token") {
                        accumulatedText += data;
                        bodyEl.innerHTML = renderMarkdown(accumulatedText);
                        scrollToBottom();
                    } else if (event === "citations") {
                        renderSegmentedSources(data);
                    } else if (event === "done") {
                        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                        timeEl.textContent = `${elapsed}s (Multi-Card)`;
                    }
                }
            }
        } catch (err) {
            console.error("Compare stream error:", err);
            bodyEl.innerHTML = `<p style="color: #EF4444;">Error generating comparison: ${escapeHtml(err.message)}</p>`;
        } finally {
            isStreaming = false;
            submitBtn.disabled = false;
            queryInput.disabled = false;
            queryInput.focus();
        }
    }

    function appendUserMessage(text, comparedCards = []) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message user-message";

        let badgesHtml = "";
        if (comparedCards && comparedCards.length > 0) {
            badgesHtml = `<div class="compared-cards-tags">` +
                comparedCards.map(name => `<span class="card-tag-badge">⚖️ ${escapeHtml(name)}</span>`).join("") +
                `</div>`;
        }

        msgDiv.innerHTML = `
            <div class="message-content">
                <div class="message-header">
                    <span class="sender-name">You</span>
                    <span class="message-time">Just now</span>
                </div>
                ${badgesHtml}
                <div class="message-body">
                    <p>${escapeHtml(text)}</p>
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
    }

    function createAssistantMessageElement(isComparison = false) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message assistant-message";
        const title = isComparison ? "Comparative Synthesis Engine" : "Policy Assistant";
        const iconColor = isComparison ? "var(--accent-cyan)" : "currentColor";

        msgDiv.innerHTML = `
            <div class="message-avatar" style="color: ${iconColor};">
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-header">
                    <span class="sender-name">${title}</span>
                    <span class="message-time">Synthesizing...</span>
                </div>
                <div class="message-body">
                    <div class="streaming-indicator">
                        <span class="pulse-dot"></span>
                        <span>${isComparison ? "Executing cross-card hybrid retrieval & synthesis..." : "Retrieving certificate clauses..."}</span>
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

    function renderSegmentedSources(citationsMap) {
        if (!citationsMap || Object.keys(citationsMap).length === 0) {
            citationsEmptyState.style.display = "flex";
            sourcesList.style.display = "none";
            sourcesCount.textContent = "0 Chunks";
            return;
        }

        citationsEmptyState.style.display = "none";
        sourcesList.style.display = "flex";
        sourcesList.innerHTML = "";

        let totalChunks = 0;
        const cardIds = Object.keys(citationsMap);

        cardIds.forEach((cid, idx) => {
            const items = citationsMap[cid] || [];
            totalChunks += items.length;

            const cardMeta = cardsData.find(c => c.card_id === cid);
            const displayName = cardMeta ? `${cardMeta.bank} — ${cardMeta.card_name}` : cid;

            const accordion = document.createElement("div");
            accordion.className = `bank-accordion ${idx === 0 ? "open" : ""}`;

            accordion.innerHTML = `
                <div class="bank-accordion-header">
                    <div class="bank-accordion-title">
                        <span class="bank-accordion-icon">▶</span>
                        <span>${escapeHtml(displayName)}</span>
                    </div>
                    <span class="rank-badge">${items.length} Sources</span>
                </div>
                <div class="bank-accordion-body"></div>
            `;

            const bodyEl = accordion.querySelector(".bank-accordion-body");
            const headerEl = accordion.querySelector(".bank-accordion-header");

            headerEl.addEventListener("click", () => {
                accordion.classList.toggle("open");
            });

            items.forEach((s, itemIdx) => {
                const card = document.createElement("div");
                card.className = "citation-card";
                card.innerHTML = `
                    <div class="citation-card-header">
                        <span class="rank-badge">#${s.rank || itemIdx + 1}</span>
                        <span class="page-badge">Page ${s.page_number}</span>
                        <span class="score-badge">Score: ${s.score || "—"}</span>
                    </div>
                    <div class="breadcrumb-trail">${escapeHtml(s.breadcrumb || "Certificate Section")}</div>
                    <div class="chunk-snippet">${escapeHtml(s.snippet || s.content || "")}</div>
                    <button type="button" class="expand-chunk-btn">View Full Section ▾</button>
                    <div class="full-chunk-content">${escapeHtml(s.full_content || s.content || s.snippet || "")}</div>
                `;

                const expandBtn = card.querySelector(".expand-chunk-btn");
                const fullContent = card.querySelector(".full-chunk-content");
                expandBtn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    const isOpen = fullContent.style.display === "block";
                    fullContent.style.display = isOpen ? "none" : "block";
                    expandBtn.textContent = isOpen ? "View Full Section ▾" : "Collapse Section ▴";
                });

                bodyEl.appendChild(card);
            });

            sourcesList.appendChild(accordion);
        });

        sourcesCount.textContent = `${totalChunks} Citations Across ${cardIds.length} Cards`;
    }

    function renderMarkdown(raw) {
        if (!raw) return "";

        // Escape HTML
        let text = raw
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Format Markdown Tables
        text = text.replace(/((?:\|[^\n]+\|\r?\n)+)/g, (match) => {
            const rawLines = match.trim().split(/\r?\n/).map(l => l.trim()).filter(l => l.startsWith("|"));
            if (rawLines.length < 2) return match;

            const headerLine = rawLines[0];
            const headerCells = headerLine.split("|").slice(1, -1).map(c => c.trim());
            const hasDelimiter = rawLines[1].includes("---");
            const dataLines = hasDelimiter ? rawLines.slice(2) : rawLines.slice(1);

            let tableHtml = '<div class="matrix-table-container"><table class="matrix-table"><thead><tr>';
            headerCells.forEach((h, colIdx) => {
                const isLast = colIdx === headerCells.length - 1;
                tableHtml += `<th class="${isLast ? 'col-winner' : ''}">${h}</th>`;
            });
            tableHtml += '</tr></thead><tbody>';

            dataLines.forEach(line => {
                const cells = line.split("|").slice(1, -1).map(c => c.trim());
                if (cells.length === 0) return;
                tableHtml += '<tr>';
                cells.forEach((c, colIdx) => {
                    const isLast = colIdx === cells.length - 1;
                    tableHtml += `<td class="${isLast ? 'col-winner' : ''}">${c}</td>`;
                });
                tableHtml += '</tr>';
            });

            tableHtml += '</tbody></table></div>';
            return tableHtml;
        });

        // Format Headers
        text = text.replace(/^###\s+(.*$)/gim, "<h3>$1</h3>");

        // Format Citations tag: [Policy Name, Section X, Page Y]
        text = text.replace(/\[([^\]]+,\s*Page\s*\d+)\]/gi, '<span class="citation-tag">[$1]</span>');
        text = text.replace(/\[(Page\s*\d+[^\]]*)\]/gi, '<span class="citation-tag">[$1]</span>');

        // Format Bold
        text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

        // Format Bullet list items
        text = text.replace(/^\s*-\s+(.*$)/gim, "<li>$1</li>");
        text = text.replace(/(<li>.*<\/li>)/gims, "<ul>$1</ul>");
        text = text.replace(/<\/ul>\s*<ul>/g, "");

        // Convert double newlines to paragraph breaks
        const paragraphs = text.split(/\n\n+/);
        return paragraphs
            .map((p) => {
                p = p.trim();
                if (p.startsWith("<h3>") || p.startsWith("<ul>") || p.startsWith("<div") || p.startsWith("<table")) {
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
