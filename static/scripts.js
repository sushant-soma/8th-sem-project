document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('viewTestDataBtn').addEventListener('click', function() {
        // Make an AJAX request to fetch the test data
        var xhr = new XMLHttpRequest();
        xhr.open('GET', '/get_test_data', true);
        xhr.onreadystatechange = function() {
            if (xhr.readyState === 4 && xhr.status === 200) {
                var testData = JSON.parse(xhr.responseText);
                displayTestData(testData);
            }
        };
        xhr.send();
    });

    document.getElementById('viewTrainingDataBtn').addEventListener('click', function() {
        // Make an AJAX request to fetch the training data (if needed)
    });
});

function displayTestData(testData) {
    var table = document.getElementById('dataset');
    // Clear existing table rows
    table.innerHTML = '';
    // Create table header
    var thead = document.createElement('thead');
    var headerRow = document.createElement('tr');
    var headers = ['Class', 'T3-resin uptake test', 'Total Serum Thyroxin', 'Total Serum Triiodothyronine', 'Basal thyroid-stimulating hormone (TSH)', 'Maximal absolute difference of TSH'];
    headers.forEach(function(headerText) {
        var th = document.createElement('th');
        th.textContent = headerText;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    // Create table body
    var tbody = document.createElement('tbody');
    testData.forEach(function(row) {
        var tr = document.createElement('tr');
        row.forEach(function(cell) {
            var td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
}
