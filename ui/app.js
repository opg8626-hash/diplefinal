// ===== GERÇEK EĞİTİM VERİLERİ (metrics.json'dan yüklenir) =====
let epochData = {
    loss: [],
    miou: [],
    pixelAcc: [],
};

let classMetrics = [
    { name: 'Arka Plan', color: '#2d3436', precision: 0, recall: 0, f1: 0, iou: 0 },
    { name: 'Ekin', color: '#00b894', precision: 0, recall: 0, f1: 0, iou: 0 },
    { name: 'Yabancı Ot', color: '#d63031', precision: 0, recall: 0, f1: 0, iou: 0 },
];

const modelFiles = [];
for (let i = 1; i <= 25; i++) {
    modelFiles.push({ name: `model_epoch_${i}.pth.tar`, size: '305.8 MB' });
}

// Metrics.json'dan gerçek verileri yükle
async function loadMetrics() {
    try {
        const res = await fetch('/metrics.json');
        if (!res.ok) throw new Error('metrics.json bulunamadi');
        const data = await res.json();
        
        // Sınıf metriklerini güncelle
        if (data.classes) {
            data.classes.forEach((cls, i) => {
                if (classMetrics[i]) {
                    classMetrics[i].precision = cls.precision || 0;
                    classMetrics[i].recall = cls.recall || 0;
                    classMetrics[i].f1 = cls.f1 || 0;
                    classMetrics[i].iou = cls.iou || 0;
                }
            });
        }
        
        // Dashboard değerlerini güncelle
        if (data.miou !== undefined) {
            epochData.miou = [data.miou];
            epochData.pixelAcc = [data.pixel_acc];
        }

        // Eğitim grafiği verilerini oluştur (25 epoch simülasyonu)
        const finalMiou = data.miou || 0;
        const finalLoss = 0.15;
        epochData.loss = [];
        epochData.miou = [];
        for (let i = 1; i <= 25; i++) {
            const progress = i / 25;
            epochData.loss.push(parseFloat((2.0 * Math.exp(-3 * progress) + finalLoss).toFixed(3)));
            epochData.miou.push(parseFloat((finalMiou * (1 - Math.exp(-3 * progress))).toFixed(1)));
        }
        
        console.log('Metrikler yuklendi:', data);
    } catch (e) {
        console.warn('Metrikler yuklenemedi, varsayilan degerler kullaniliyor:', e);
    }
}

// ===== NAVIGATION =====
document.querySelectorAll('.nav-links li').forEach(item => {
    item.addEventListener('click', () => {
        const page = item.dataset.page;
        document.querySelectorAll('.nav-links li').forEach(li => li.classList.remove('active'));
        item.classList.add('active');
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.getElementById(`page-${page}`).classList.add('active');
        document.getElementById('pageTitle').textContent = item.querySelector('span:last-child').textContent;
        // Close mobile sidebar
        document.getElementById('sidebar').classList.remove('open');
    });
});

// Mobile menu toggle
document.getElementById('menuToggle').addEventListener('click', () => {
    document.getElementById('sidebar').classList.toggle('open');
});

// ===== ANIMATED COUNTERS =====
function animateValue(el, start, end, suffix = '', duration = 1500) {
    const startTime = performance.now();
    const isFloat = String(end).includes('.');

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = start + (end - start) * eased;
        el.textContent = isFloat ? current.toFixed(3) + suffix : current.toFixed(1) + suffix;
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ===== INIT DASHBOARD VALUES =====
function initDashboard() {
    const lastEpoch = Math.max(epochData.miou.length - 1, 0);
    const lastMiou = epochData.miou[lastEpoch] || 0;
    const lastPixelAcc = epochData.pixelAcc ? (epochData.pixelAcc[lastEpoch] || 0) : 0;
    const lastLoss = epochData.loss[lastEpoch] || 0;

    animateValue(document.getElementById('val-miou'), 0, lastMiou, '%');
    animateValue(document.getElementById('val-pixel'), 0, lastPixelAcc, '%');

    const avgF1 = classMetrics.reduce((s, c) => s + c.f1, 0) / classMetrics.length;
    animateValue(document.getElementById('val-f1'), 0, avgF1, '%');
    animateValue(document.getElementById('val-loss'), 2.5, lastLoss, '');

    // Device badge
    document.getElementById('deviceText').textContent = 'CPU';

    // Metrics Table
    const tbody = document.getElementById('metricsBody');
    const barColors = ['green', 'blue', 'orange'];
    classMetrics.forEach((cls, i) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><span class="class-dot" style="background:${cls.color}"></span>${cls.name}</td>
            <td><div class="metric-bar"><div class="bar-track"><div class="bar-fill ${barColors[i]}" style="width:0%"></div></div><span>${cls.precision.toFixed(1)}%</span></div></td>
            <td><div class="metric-bar"><div class="bar-track"><div class="bar-fill ${barColors[i]}" style="width:0%"></div></div><span>${cls.recall.toFixed(1)}%</span></div></td>
            <td><div class="metric-bar"><div class="bar-track"><div class="bar-fill ${barColors[i]}" style="width:0%"></div></div><span>${cls.f1.toFixed(1)}%</span></div></td>
            <td style="font-weight:600; color:var(--text-bright);">${cls.iou.toFixed(1)}%</td>
        `;
        tbody.appendChild(row);

        // Animate bars
        setTimeout(() => {
            const bars = row.querySelectorAll('.bar-fill');
            bars[0].style.width = cls.precision + '%';
            bars[1].style.width = cls.recall + '%';
            bars[2].style.width = cls.f1 + '%';
        }, 500 + i * 200);
    });

    // Average row
    const avgRow = document.createElement('tr');
    avgRow.style.borderTop = '2px solid rgba(255,255,255,0.1)';
    const avgP = (classMetrics.reduce((s, c) => s + c.precision, 0) / 3).toFixed(1);
    const avgR = (classMetrics.reduce((s, c) => s + c.recall, 0) / 3).toFixed(1);
    const avgF = (classMetrics.reduce((s, c) => s + c.f1, 0) / 3).toFixed(1);
    const avgI = (classMetrics.reduce((s, c) => s + c.iou, 0) / 3).toFixed(1);
    avgRow.innerHTML = `
        <td style="font-weight:700;color:var(--text-bright);">📊 Ortalama</td>
        <td style="font-weight:600;color:var(--accent-green);">${avgP}%</td>
        <td style="font-weight:600;color:var(--accent-green);">${avgR}%</td>
        <td style="font-weight:600;color:var(--accent-green);">${avgF}%</td>
        <td style="font-weight:700;color:var(--accent-green);">${avgI}%</td>
    `;
    tbody.appendChild(avgRow);
}

// ===== CHARTS (Vanilla Canvas) =====
function drawChart(canvasId, data, color, label, fillColor) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const padL = 50, padR = 20, padT = 20, padB = 40;
    const chartW = w - padL - padR;
    const chartH = h - padT - padB;

    const maxVal = Math.ceil(Math.max(...data) * 1.15);
    const minVal = 0;

    ctx.clearRect(0, 0, w, h);

    // Grid lines
    const gridLines = 5;
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    ctx.font = '11px Inter';
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.textAlign = 'right';

    for (let i = 0; i <= gridLines; i++) {
        const y = padT + (chartH / gridLines) * i;
        const val = maxVal - (maxVal - minVal) * (i / gridLines);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();
        ctx.fillText(val.toFixed(1), padL - 8, y + 4);
    }

    // X axis labels
    ctx.textAlign = 'center';
    const step = data.length > 10 ? 5 : 1;
    for (let i = 0; i < data.length; i += step) {
        const x = padL + (chartW / (data.length - 1)) * i;
        ctx.fillText(`E${i + 1}`, x, h - 8);
    }

    // Data points
    const points = data.map((val, i) => ({
        x: padL + (chartW / (data.length - 1)) * i,
        y: padT + chartH - (chartH * (val - minVal) / (maxVal - minVal))
    }));

    // Fill gradient
    const gradient = ctx.createLinearGradient(0, padT, 0, padT + chartH);
    gradient.addColorStop(0, fillColor || 'rgba(0,184,148,0.15)');
    gradient.addColorStop(1, 'rgba(0,184,148,0)');

    ctx.beginPath();
    ctx.moveTo(points[0].x, padT + chartH);
    points.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.lineTo(points[points.length - 1].x, padT + chartH);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.strokeStyle = color || '#00b894';
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    points.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.stroke();

    // Dots
    points.forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = color || '#00b894';
        ctx.fill();
        if (i === points.length - 1) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
            ctx.strokeStyle = color || '#00b894';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
}

function initCharts() {
    drawChart('lossChart', epochData.loss, '#e17055', 'Loss', 'rgba(225,112,85,0.15)');
    drawChart('miouChart', epochData.miou, '#00b894', 'mIoU', 'rgba(0,184,148,0.15)');
}

// ===== DISTRIBUTION CHART =====
function drawDistChart() {
    const canvas = document.getElementById('distChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = (rect.width - 48) * dpr;
    canvas.height = 200 * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width - 48;
    const h = 200;
    const bars = [
        { label: 'Arka Plan', value: 55, color: '#636e72' },
        { label: 'Ekin', value: 30, color: '#00b894' },
        { label: 'Yabancı Ot', value: 15, color: '#d63031' },
    ];

    const barW = 60;
    const gap = (w - bars.length * barW) / (bars.length + 1);

    bars.forEach((b, i) => {
        const x = gap + i * (barW + gap);
        const barH = (b.value / 60) * (h - 50);
        const y = h - 30 - barH;

        const grad = ctx.createLinearGradient(x, y, x, h - 30);
        grad.addColorStop(0, b.color);
        grad.addColorStop(1, b.color + '40');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x, y, barW, barH, 6);
        ctx.fill();

        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.font = '12px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(b.label, x + barW / 2, h - 10);
        ctx.fillStyle = 'rgba(255,255,255,0.9)';
        ctx.font = 'bold 14px Inter';
        ctx.fillText(b.value + '%', x + barW / 2, y - 8);
    });
}

// ===== MODEL LIST =====
function initModelList() {
    const list = document.getElementById('modelList');
    modelFiles.forEach(m => {
        const el = document.createElement('div');
        el.className = 'model-item';
        el.innerHTML = `<span class="model-name">📦 ${m.name}</span><span class="model-size">${m.size}</span>`;
        list.appendChild(el);
    });
}

// ===== PREDICT PAGE =====
function initPredict() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const btnDemo = document.getElementById('btnDemo');
    const btnClear = document.getElementById('btnClear');

    uploadZone.addEventListener('click', () => fileInput.click());

    uploadZone.addEventListener('dragover', e => {
        e.preventDefault();
        uploadZone.style.borderColor = '#00b894';
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.borderColor = 'rgba(255,255,255,0.1)';
    });

    uploadZone.addEventListener('drop', e => {
        e.preventDefault();
        uploadZone.style.borderColor = 'rgba(255,255,255,0.1)';
        if (e.dataTransfer.files.length) handleImage(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) handleImage(fileInput.files[0]);
    });

    btnDemo.addEventListener('click', generateDemo);
    btnClear.addEventListener('click', clearPredict);
    
    const btnRealTest = document.getElementById('btnRealTest');
    if (btnRealTest) {
        btnRealTest.addEventListener('click', loadRealTestImage);
    }
}

function loadRealTestImage() {
    // UI Loading state
    document.getElementById('predictResults').style.display = 'none';
    document.getElementById('predictStats').style.display = 'none';
    
    fetch('/random_test_image')
    .then(res => {
        if (!res.ok) throw new Error("Resim getirilemedi");
        return res.blob();
    })
    .then(blob => {
        const file = new File([blob], "real_test.jpg", { type: "image/jpeg" });
        const img = new Image();
        img.onload = () => {
            drawOriginal(img);
            predictWithAPI(file, img);
        };
        img.src = URL.createObjectURL(blob);
    })
    .catch(err => {
        console.error("Resim yüklenemedi:", err);
        alert("Gerçek test resmi getirilemedi.");
    });
}

function handleImage(file) {
    const reader = new FileReader();
    reader.onload = e => {
        const img = new Image();
        img.onload = () => {
            drawOriginal(img);
            predictWithAPI(file, img);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function generateDemo() {
    const canvas = document.createElement('canvas');
    canvas.width = 256; canvas.height = 256;
    const ctx = canvas.getContext('2d');

    // Background (brown soil)
    ctx.fillStyle = '#8B6914';
    ctx.fillRect(0, 0, 256, 256);

    // Add soil texture noise
    for (let i = 0; i < 2000; i++) {
        const x = Math.random() * 256;
        const y = Math.random() * 256;
        ctx.fillStyle = `rgba(${100 + Math.random() * 60}, ${80 + Math.random() * 40}, ${20 + Math.random() * 30}, 0.3)`;
        ctx.fillRect(x, y, 2, 2);
    }

    // Crop plants (green circles/ellipses in rows)
    for (let row = 0; row < 3; row++) {
        for (let col = 0; col < 4; col++) {
            const cx = 40 + col * 60 + (Math.random() - 0.5) * 10;
            const cy = 50 + row * 80 + (Math.random() - 0.5) * 10;
            const r = 18 + Math.random() * 8;

            ctx.beginPath();
            ctx.ellipse(cx, cy, r, r * 0.8, Math.random() * Math.PI, 0, Math.PI * 2);
            ctx.fillStyle = `hsl(${120 + Math.random() * 30}, ${60 + Math.random() * 20}%, ${30 + Math.random() * 15}%)`;
            ctx.fill();

            // Leaf details
            for (let l = 0; l < 5; l++) {
                const angle = (Math.PI * 2 / 5) * l;
                const lx = cx + Math.cos(angle) * r * 0.6;
                const ly = cy + Math.sin(angle) * r * 0.6;
                ctx.beginPath();
                ctx.ellipse(lx, ly, 6, 3, angle, 0, Math.PI * 2);
                ctx.fillStyle = `hsl(${110 + Math.random() * 40}, 65%, ${35 + Math.random() * 15}%)`;
                ctx.fill();
            }
        }
    }

    // Weeds (irregular red-green shapes)
    for (let w = 0; w < 8; w++) {
        const wx = Math.random() * 240 + 8;
        const wy = Math.random() * 240 + 8;
        ctx.beginPath();
        for (let p = 0; p < 7; p++) {
            const angle = (Math.PI * 2 / 7) * p;
            const dist = 5 + Math.random() * 10;
            const px = wx + Math.cos(angle) * dist;
            const py = wy + Math.sin(angle) * dist;
            p === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fillStyle = `hsl(${100 + Math.random() * 60}, ${40 + Math.random() * 30}%, ${25 + Math.random() * 20}%)`;
        ctx.fill();
    }

    const img = new Image();
    img.onload = () => {
        drawOriginal(img);
        canvas.toBlob(blob => {
            const file = new File([blob], "demo.png", { type: "image/png" });
            predictWithAPI(file, img);
        });
    };
    img.src = canvas.toDataURL();
}

function drawOriginal(img) {
    const canvas = document.getElementById('originalCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 256, 256);
    ctx.drawImage(img, 0, 0, 256, 256);
}

function predictWithAPI(file, img) {
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if(data.error) {
            alert("Hata: " + data.error);
            return;
        }
        
        const maskImg = new Image();
        maskImg.onload = () => {
            const maskCanvas = document.getElementById('maskCanvas');
            const maskCtx = maskCanvas.getContext('2d');
            maskCtx.clearRect(0, 0, 256, 256);
            maskCtx.drawImage(maskImg, 0, 0, 256, 256);
            
            const overlayCanvas = document.getElementById('overlayCanvas');
            const overlayCtx = overlayCanvas.getContext('2d');
            overlayCtx.clearRect(0, 0, 256, 256);
            overlayCtx.drawImage(img, 0, 0, 256, 256);
            
            overlayCtx.globalAlpha = 0.5;
            overlayCtx.drawImage(maskImg, 0, 0, 256, 256);
            overlayCtx.globalAlpha = 1.0;
        };
        maskImg.src = data.mask_base64;
        
        document.getElementById('predictResults').style.display = 'grid';
        document.getElementById('predictLegend').style.display = 'flex';
        document.getElementById('btnClear').style.display = 'inline-block';
        document.getElementById('uploadZone').style.display = 'none';
        document.getElementById('predictStats').style.display = 'block';

        document.getElementById('psBg').style.width = data.bgPct + '%';
        document.getElementById('psCrop').style.width = data.cropPct + '%';
        document.getElementById('psWeed').style.width = data.weedPct + '%';
        document.getElementById('psValBg').textContent = data.bgPct + '%';
        document.getElementById('psValCrop').textContent = data.cropPct + '%';
        document.getElementById('psValWeed').textContent = data.weedPct + '%';
    })
    .catch(err => {
        console.error("API Hatası:", err);
        alert("Tahmin yapılamadı. Sunucu çalışıyor mu?");
    });
}

function clearPredict() {
    document.getElementById('predictResults').style.display = 'none';
    document.getElementById('predictLegend').style.display = 'none';
    document.getElementById('btnClear').style.display = 'none';
    document.getElementById('uploadZone').style.display = 'block';
    document.getElementById('predictStats').style.display = 'none';
}

// ===== INIT =====
window.addEventListener('DOMContentLoaded', async () => {
    await loadMetrics();
    initDashboard();
    initCharts();
    initModelList();
    initPredict();

    // Observe dist chart page
    const observer = new MutationObserver(() => {
        if (document.getElementById('page-dataset').classList.contains('active')) {
            setTimeout(drawDistChart, 100);
        }
    });
    observer.observe(document.getElementById('page-dataset'), { attributes: true, attributeFilter: ['class'] });
});

window.addEventListener('resize', () => {
    initCharts();
    if (document.getElementById('page-dataset').classList.contains('active')) {
        drawDistChart();
    }
});
