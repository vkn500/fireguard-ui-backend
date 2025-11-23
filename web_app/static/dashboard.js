function loadCharts(labelCounts, severityCounts, dailyCounts) {

    // ---------------------------
    // Fire vs Smoke Pie Chart
    // ---------------------------
    new Chart(document.getElementById("labelChart"), {
        type: "pie",
        data: {
            labels: labelCounts.labels,
            datasets: [{
                data: labelCounts.values,
                backgroundColor: ["#ff0000", "#ffaa00"]
            }]
        }
    });

    // ---------------------------
    // Severity Bar Chart
    // ---------------------------
    new Chart(document.getElementById("severityChart"), {
        type: "bar",
        data: {
            labels: ["SAFE", "SMOKE", "SMALL FIRE", "LARGE FIRE"],
            datasets: [{
                label: "Count",
                data: severityCounts,
                backgroundColor: ["#00cc00", "#cccc00", "#ff8800", "#ff0000"]
            }]
        }
    });

    // ---------------------------
    // Daily Alerts Line Chart
    // ---------------------------
    new Chart(document.getElementById("dailyChart"), {
        type: "line",
        data: {
            labels: dailyCounts.labels,
            datasets: [{
                label: "Alerts",
                data: dailyCounts.values,
                borderColor: "#ff0000",
                backgroundColor: "rgba(255, 0, 0, 0.3)",
                fill: true
            }]
        }
    });
}
