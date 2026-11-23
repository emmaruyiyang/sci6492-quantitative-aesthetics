const form = document.querySelector("#query-form");
const results = document.querySelector("#results");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const word = document.querySelector("#word").value;
  const topN = document.querySelector("#top_n").value;
  results.textContent = "Loading\u2026";

  const response = await fetch(`/api/similar?word=${encodeURIComponent(word)}&top_n=${topN}`);
  const data = await response.json();

  if (!response.ok) {
    results.innerHTML = `<p class="error"></p>`;
    results.querySelector("p").textContent = data.error;
    return;
  }

  const rows = data.results.map((item, index) => `
    <tr>
      <td class="rank">${index + 1}</td>
      <td>${item.word}</td>
      <td class="score">${item.similarity.toFixed(4)}</td>
    </tr>`).join("");

  results.innerHTML = `<table>${rows}</table>`;
});
