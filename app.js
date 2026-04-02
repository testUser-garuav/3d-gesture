// ═══════════════════════════════════════════════════════════════════
//  3D Particle System with Hand Gesture Control
// ═══════════════════════════════════════════════════════════════════

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { HandLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs';

const PARTICLE_COUNT = 8000;

// ─── Palettes ────────────────────────────────────────────────────
const PALETTES = [
  ['#ff6fd8','#ff9a3c','#ff3cac','#ffd700'],
  ['#00c9ff','#0052d4','#4facfe','#00f2fe'],
  ['#11998e','#38ef7d','#a8e063','#56ab2f'],
  ['#f72585','#7209b7','#3a0ca3','#4cc9f0'],
  ['#ff0000','#ff4500','#ff8c00','#ffd700'],
  ['#fbc2eb','#a6c1ee','#ffecd2','#c3cfe2'],
  ['#667eea','#764ba2','#f093fb','#4facfe'],
].map(p => p.map(c => new THREE.Color(c)));

// ─── Templates ───────────────────────────────────────────────────
const TEMPLATES = {
  heart: { icon: '\u2764', name: 'Heart', gen(i,n) {
    const t=(i/n)*Math.PI*2, x=16*Math.pow(Math.sin(t),3), y=13*Math.cos(t)-5*Math.cos(2*t)-2*Math.cos(3*t)-Math.cos(4*t);
    return [x*0.18+(Math.random()-0.5)*0.3, y*0.18+(Math.random()-0.5)*0.3, (Math.random()-0.5)*3];
  }},
  flower: { icon: '\uD83C\uDF38', name: 'Flower', gen(i,n) {
    const t=(i/n)*Math.PI*2, r=3*Math.cos(6*t)+Math.random()*0.5;
    return [r*Math.cos(t), r*Math.sin(t), (Math.random()-0.5)*1.5];
  }},
  saturn: { icon: '\uD83E\uDE90', name: 'Saturn', gen(i,n) {
    if(i<n*0.6){const p=Math.acos(2*Math.random()-1),t=Math.random()*Math.PI*2,r=2+(Math.random()-0.5)*0.3;return[r*Math.sin(p)*Math.cos(t),r*Math.sin(p)*Math.sin(t),r*Math.cos(p)];}
    const t=Math.random()*Math.PI*2,r=3.5+Math.random()*1.5;return[r*Math.cos(t),(Math.random()-0.5)*0.2,r*Math.sin(t)];
  }},
  firework: { icon: '\uD83C\uDF86', name: 'Firework', gen(i,n) {
    const b=Math.floor(Math.random()*5),cx=(b-2)*2.5,cy=Math.sin(b*1.3)*2,p=Math.acos(2*Math.random()-1),t=Math.random()*Math.PI*2,r=1.2+Math.random();
    return[cx+r*Math.sin(p)*Math.cos(t),cy+r*Math.sin(p)*Math.sin(t),r*Math.cos(p)];
  }},
  star: { icon: '\u2B50', name: 'Star', gen(i,n) {
    const t=(i/n)*Math.PI*2,pts=5,idx=Math.floor((t/(Math.PI*2))*(pts*2)),r1=3.5,r2=1.5;
    const rC=idx%2===0?r1:r2,rN=idx%2===0?r2:r1,bl=(t%(Math.PI/pts))/(Math.PI/pts),rF=rC+(rN-rC)*bl+(Math.random()-0.5)*0.4;
    return[rF*Math.cos(t),rF*Math.sin(t),(Math.random()-0.5)*1.5];
  }},
  spiral: { icon: '\uD83C\uDF00', name: 'Spiral', gen(i,n) {
    const arms=3,arm=i%arms,t=(i/n)*6,a=t*Math.PI*2+(arm*Math.PI*2/arms),r=t*0.8,sp=0.15+t*0.08;
    return[r*Math.cos(a)+(Math.random()-0.5)*sp,(Math.random()-0.5)*(0.3+t*0.05),r*Math.sin(a)+(Math.random()-0.5)*sp];
  }},
  dna: { icon: '\uD83E\uDDEC', name: 'DNA', gen(i,n) {
    const t=(i/n)*Math.PI*8-Math.PI*4,s=i%2,a=t+s*Math.PI;
    return[2*Math.cos(a)+(Math.random()-0.5)*0.3,t*0.5,2*Math.sin(a)+(Math.random()-0.5)*0.3];
  }},
  butterfly: { icon: '\uD83E\uDD8B', name: 'Butterfly', gen(i,n) {
    const t=(i/n)*Math.PI*2,r=Math.exp(Math.cos(t))-2*Math.cos(4*t)+Math.pow(Math.sin(t/12),5);
    return[r*Math.sin(t)*1.5+(Math.random()-0.5)*0.3,r*Math.cos(t)*1.5+(Math.random()-0.5)*0.3,(Math.random()-0.5)];
  }},
  sphere: { icon: '\uD83D\uDD2E', name: 'Sphere', gen(i,n) {
    const p=Math.acos(2*Math.random()-1),t=Math.random()*Math.PI*2;
    return[3*Math.sin(p)*Math.cos(t),3*Math.sin(p)*Math.sin(t),3*Math.cos(p)];
  }},
};
const TKEYS = Object.keys(TEMPLATES);

// ═══════════════════════════════════════════════════════════════════
//  State
// ═══════════════════════════════════════════════════════════════════
let currentTemplate = 0;
let currentPalette = 0;
let gestureSmoothed = 'none';
let gestureHoldTime = 0;
let lastRawGesture = 'none';
let handDetected = false;
let handPos = { x: 0, y: 0 };
let keyboardOverrideUntil = 0;

// Particle interaction state — these directly drive the visuals
let expandFactor = 0;    // -1 (contracted) to 0 (rest) to +1 (expanded)
let attractActive = false;
let attractX = 0, attractY = 0, attractZ = 0;
let fireworkTimer = 0;   // >0 means firework is active

// ═══════════════════════════════════════════════════════════════════
//  Three.js
// ═══════════════════════════════════════════════════════════════════
const canvas = document.getElementById('canvas3d');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();

const cam = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
cam.position.set(0, 2, 25);

const orbitCtrl = new OrbitControls(cam, canvas);
orbitCtrl.enableDamping = true;
orbitCtrl.dampingFactor = 0.05;
orbitCtrl.autoRotate = true;
orbitCtrl.autoRotateSpeed = 0.5;

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, cam));
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.8, 0.4, 0.85);
composer.addPass(bloom);

// ═══════════════════════════════════════════════════════════════════
//  Particles
// ═══════════════════════════════════════════════════════════════════
const geo = new THREE.BufferGeometry();
const pos = new Float32Array(PARTICLE_COUNT * 3);    // current positions
const col = new Float32Array(PARTICLE_COUNT * 3);    // current colors
const siz = new Float32Array(PARTICLE_COUNT);        // sizes
const rest = new Float32Array(PARTICLE_COUNT * 3);   // rest (template) positions
const vel = new Float32Array(PARTICLE_COUNT * 3);    // velocities (for firework)
const colTarget = new Float32Array(PARTICLE_COUNT * 3);
const phase = new Float32Array(PARTICLE_COUNT);

for (let i = 0; i < PARTICLE_COUNT; i++) {
  phase[i] = Math.random() * Math.PI * 2;
  siz[i] = 0.5 + Math.random() * 1.5;
}

// Generate rest positions + colors for a template/palette
function generateTemplate(tIdx, pIdx, scatter = true) {
  const tmpl = TEMPLATES[TKEYS[tIdx]];
  const pal = PALETTES[pIdx];
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const [x,y,z] = tmpl.gen(i, PARTICLE_COUNT);
    const i3 = i*3;
    rest[i3]=x; rest[i3+1]=y; rest[i3+2]=z;
    const c = pal[i % pal.length];
    colTarget[i3]=c.r; colTarget[i3+1]=c.g; colTarget[i3+2]=c.b;

    // Scatter particles outward briefly on template change for dramatic transition
    if (scatter) {
      const angle = Math.random() * Math.PI * 2;
      const dist = 1.5 + Math.random() * 2.0;
      pos[i3]   += Math.cos(angle) * dist;
      pos[i3+1] += (Math.random()-0.5) * dist;
      pos[i3+2] += Math.sin(angle) * dist;
    }
  }
  if (scatter) bloom.strength = 1.2;
}

// Initialize
function initParticles() {
  generateTemplate(currentTemplate, currentPalette, false);
  for (let i = 0; i < PARTICLE_COUNT*3; i++) {
    pos[i] = rest[i];
    col[i] = colTarget[i];
    vel[i] = 0;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('size', new THREE.BufferAttribute(siz, 1));
}

const mat = new THREE.ShaderMaterial({
  uniforms: {
    uPR: { value: renderer.getPixelRatio() },
  },
  vertexShader: `
    attribute float size;
    varying vec3 vColor;
    varying float vDist;
    uniform float uPR;
    void main() {
      vColor = color;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vDist = -mv.z;
      gl_PointSize = size * uPR * (200.0 / -mv.z);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    varying vec3 vColor;
    varying float vDist;
    void main() {
      vec2 center = gl_PointCoord - 0.5;
      float dist = length(center);
      if (dist > 0.5) discard;
      float alpha = smoothstep(0.5, 0.1, dist);
      float glow = exp(-dist * 3.0) * 0.6;
      vec3 col = vColor + glow;
      gl_FragColor = vec4(col, alpha * 0.85);
    }`,
  transparent: true, depthWrite: false,
  blending: THREE.AdditiveBlending, vertexColors: true,
});

const particles = new THREE.Points(geo, mat);
scene.add(particles);

// Stars
const sg = new THREE.BufferGeometry();
const sp = new Float32Array(2000*3);
for(let i=0;i<2000;i++){sp[i*3]=(Math.random()-0.5)*80;sp[i*3+1]=(Math.random()-0.5)*80;sp[i*3+2]=(Math.random()-0.5)*80;}
sg.setAttribute('position',new THREE.BufferAttribute(sp,3));
scene.add(new THREE.Points(sg, new THREE.PointsMaterial({color:0xffffff,size:0.08,transparent:true,opacity:0.6})));

initParticles();

// ═══════════════════════════════════════════════════════════════════
//  Actions (called by buttons, keys, and gestures)
// ═══════════════════════════════════════════════════════════════════
function doNextTemplate() {
  currentTemplate = (currentTemplate + 1) % TKEYS.length;
  generateTemplate(currentTemplate, currentPalette);
  updateUI();
  log('Template: ' + TEMPLATES[TKEYS[currentTemplate]].name);
}

function doPrevTemplate() {
  currentTemplate = (currentTemplate - 1 + TKEYS.length) % TKEYS.length;
  generateTemplate(currentTemplate, currentPalette);
  updateUI();
}

function doSetTemplate(idx) {
  if (idx === currentTemplate || idx >= TKEYS.length) return;
  currentTemplate = idx;
  generateTemplate(currentTemplate, currentPalette);
  updateUI();
}

function doCycleColor() {
  currentPalette = (currentPalette + 1) % PALETTES.length;
  const pal = PALETTES[currentPalette];
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const c = pal[i % pal.length];
    colTarget[i*3]=c.r; colTarget[i*3+1]=c.g; colTarget[i*3+2]=c.b;
    siz[i] = 0.5 + Math.random() * 0.5;
  }
  bloom.strength = 1.2;
  log('Palette: ' + currentPalette);
}

function doFirework() {
  fireworkTimer = 180;
  // Explode particles outward from multiple burst centers
  const burstCount = 3 + Math.floor(Math.random() * 3);
  const centers = [];
  for (let b = 0; b < burstCount; b++) {
    centers.push({
      x: (Math.random() - 0.5) * 6,
      y: (Math.random() - 0.5) * 4,
      z: (Math.random() - 0.5) * 4,
    });
  }
  const pal = PALETTES[currentPalette];
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const burst = centers[i % burstCount];
    const a1 = Math.random() * Math.PI * 2, a2 = Math.acos(2 * Math.random() - 1);
    const spd = 0.3 + Math.random() * 0.8;
    vel[i*3]   = spd * Math.sin(a2) * Math.cos(a1);
    vel[i*3+1] = spd * Math.sin(a2) * Math.sin(a1) + Math.random() * 0.3;
    vel[i*3+2] = spd * Math.cos(a2);
    // Move to burst center
    pos[i*3]   = burst.x + (Math.random() - 0.5) * 1.0;
    pos[i*3+1] = burst.y + (Math.random() - 0.5) * 1.0;
    pos[i*3+2] = burst.z + (Math.random() - 0.5) * 1.0;
    // Bright palette colors
    const c = pal[i % pal.length];
    col[i*3] = c.r; col[i*3+1] = c.g; col[i*3+2] = c.b;
    siz[i] = 0.5 + Math.random() * 0.8;
  }
  bloom.strength = 1.5;
  log('FIREWORK!');
}

function doExpand(active) {
  if (active) { expandFactor = 1; bloom.strength = 1.2; log('Expanding'); }
  else { expandFactor = 0; }
}

function doContract(active) {
  if (active) { expandFactor = -1; bloom.strength = 1.2; log('Contracting'); }
  else { expandFactor = 0; }
}

function doAttract(active, x, y) {
  attractActive = active;
  if (active) { attractX = (x||0)*8; attractY = (y||0)*8; attractZ = 0; bloom.strength = 1.0; log('Attracting'); }
}

function doReset() {
  expandFactor = 0;
  attractActive = false;
  fireworkTimer = 0;
  gestureSmoothed = 'none';
  for (let i = 0; i < PARTICLE_COUNT*3; i++) vel[i] = 0;
  log('Reset');
}

function log(msg) {
  console.log('[Action] ' + msg);
}

// ═══════════════════════════════════════════════════════════════════
//  UI: template bar + action buttons
// ═══════════════════════════════════════════════════════════════════
const templateBar = document.getElementById('template-bar');
TKEYS.forEach((key, idx) => {
  const btn = document.createElement('button');
  btn.className = 'tmpl-btn' + (idx===0?' active':'');
  btn.textContent = TEMPLATES[key].icon;
  btn.title = TEMPLATES[key].name;
  btn.addEventListener('click', () => doSetTemplate(idx));
  templateBar.appendChild(btn);
});

function updateUI() {
  document.querySelectorAll('.tmpl-btn').forEach((b,i) => b.classList.toggle('active', i===currentTemplate));
  document.getElementById('template-label').textContent = 'Template: ' + TEMPLATES[TKEYS[currentTemplate]].name;
}

// Action buttons — toggle on click with auto-reset after 2 seconds
let expandTimer = 0, contractTimer = 0, attractTimer = 0;

document.getElementById('btn-firework').addEventListener('click', doFirework);
document.getElementById('btn-color').addEventListener('click', doCycleColor);
document.getElementById('btn-reset').addEventListener('click', doReset);

// Expand: click to toggle for 2 seconds
document.getElementById('btn-expand').addEventListener('click', () => {
  doExpand(true);
  clearTimeout(expandTimer);
  expandTimer = setTimeout(() => doExpand(false), 2000);
});

// Contract: click to toggle for 2 seconds
document.getElementById('btn-contract').addEventListener('click', () => {
  doContract(true);
  clearTimeout(contractTimer);
  contractTimer = setTimeout(() => doContract(false), 2000);
});

// Attract: click to toggle for 2 seconds
document.getElementById('btn-attract').addEventListener('click', () => {
  doAttract(true, 0, 0);
  clearTimeout(attractTimer);
  attractTimer = setTimeout(() => doAttract(false), 2000);
});

// ═══════════════════════════════════════════════════════════════════
//  Animation — simple and direct
// ═══════════════════════════════════════════════════════════════════
const clock = new THREE.Clock();
let frame = 0;

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const t = clock.getElapsedTime();
  frame++;

  // Firework countdown
  const isFirework = fireworkTimer > 0;
  if (isFirework) fireworkTimer--;

  // Bloom — keep subtle, never overpowering
  if (isFirework) bloom.strength += (1.2 - bloom.strength) * 0.05;
  else if (expandFactor !== 0 || attractActive) bloom.strength += (1.0 - bloom.strength) * 0.05;
  else bloom.strength += (0.8 - bloom.strength) * 0.03;

  // Per-particle update
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const i3 = i*3;
    const rx = rest[i3], ry = rest[i3+1], rz = rest[i3+2];

    if (isFirework) {
      // ── FIREWORK MODE: dramatic explosion with trails ──
      vel[i3+1] -= 0.008; // stronger gravity
      vel[i3]   *= 0.985;
      vel[i3+1] *= 0.985;
      vel[i3+2] *= 0.985;
      pos[i3]   += vel[i3];
      pos[i3+1] += vel[i3+1];
      pos[i3+2] += vel[i3+2];

      // Sparkle colors during firework
      if (fireworkTimer > 100) {
        const sparkle = Math.random();
        if (sparkle > 0.95) {
          col[i3] = 1.0; col[i3+1] = 1.0; col[i3+2] = 1.0; // white flash
        }
      }
      // Dynamic size during firework — shrink as they fade
      siz[i] = (fireworkTimer / 200) * (0.5 + Math.random() * 1.5);
    } else {
      // ── NORMAL MODE: lerp toward target ──

      // Compute target based on expand/contract
      let tx = rx, ty = ry, tz = rz;

      if (expandFactor > 0.01) {
        // Expand: scale outward by 8x for dramatic spread
        const scale = 1.0 + expandFactor * 7.0;
        tx = rx * scale;
        ty = ry * scale;
        tz = rz * scale;
      } else if (expandFactor < -0.01) {
        // Contract: collapse to tiny core
        const scale = Math.max(0.02, 1.0 + expandFactor * 0.98);
        tx = rx * scale;
        ty = ry * scale;
        tz = rz * scale;
      }

      // Attract override: pull all particles to a single point
      if (attractActive) {
        const jitter = 0.3;
        tx = attractX + (Math.random()-0.5)*jitter;
        ty = attractY + (Math.random()-0.5)*jitter;
        tz = attractZ + (Math.random()-0.5)*jitter;
      }

      // Faster lerp for snappy response
      const lerpSpeed = attractActive ? 0.15 : 0.12;
      pos[i3]   += (tx - pos[i3])   * lerpSpeed;
      pos[i3+1] += (ty - pos[i3+1]) * lerpSpeed;
      pos[i3+2] += (tz - pos[i3+2]) * lerpSpeed;

      // Floating motion
      const ph = phase[i];
      pos[i3]   += Math.sin(t*0.7+ph) * 0.005;
      pos[i3+1] += Math.cos(t*0.5+ph*1.3) * 0.005;
      pos[i3+2] += Math.sin(t*0.6+ph*0.7) * 0.005;

      // Size reacts to state
      if (expandFactor > 0.01) {
        siz[i] = 0.3 + Math.sin(t*3+phase[i])*0.2; // smaller when spread
      } else if (expandFactor < -0.01) {
        siz[i] = 1.0 + Math.sin(t*4+phase[i])*0.5; // bigger when contracted (bright core)
      } else if (attractActive) {
        siz[i] = 1.0 + Math.sin(t*5+phase[i])*0.8;
      } else {
        siz[i] = 0.5 + Math.sin(t*2+phase[i])*0.3;
      }
    }

    // Fast color lerp
    col[i3]   += (colTarget[i3]   - col[i3])   * 0.15;
    col[i3+1] += (colTarget[i3+1] - col[i3+1]) * 0.15;
    col[i3+2] += (colTarget[i3+2] - col[i3+2]) * 0.15;
  }

  geo.attributes.position.needsUpdate = true;
  geo.attributes.color.needsUpdate = true;
  geo.attributes.size.needsUpdate = true;

  particles.rotation.y += dt * 0.05;
  orbitCtrl.update();
  composer.render();

}

// ═══════════════════════════════════════════════════════════════════
//  Gesture Recognition
// ═══════════════════════════════════════════════════════════════════
function dist2D(a,b){return Math.sqrt((a.x-b.x)**2+(a.y-b.y)**2);}

function recognizeGesture(lm) {
  const wrist=lm[0], thumbTip=lm[4], thumbIp=lm[3], indexTip=lm[8], indexPip=lm[6], indexMcp=lm[5];
  const middleTip=lm[12], middlePip=lm[10];
  const ringTip=lm[16], ringPip=lm[14], pinkyTip=lm[20], pinkyPip=lm[18];

  // Finger extended = tip clearly above pip (stricter margin)
  const margin = 0.02;
  const iE = indexTip.y < indexPip.y - margin;
  const mE = middleTip.y < middlePip.y - margin;
  const rE = ringTip.y < ringPip.y - margin;
  const pE = pinkyTip.y < pinkyPip.y - margin;

  // Finger clearly curled = tip below pip
  const iC = indexTip.y > indexPip.y + margin;
  const mC = middleTip.y > middlePip.y + margin;
  const rC = ringTip.y > ringPip.y + margin;
  const pC = pinkyTip.y > pinkyPip.y + margin;

  const thE = dist2D(thumbTip, wrist) > dist2D(thumbIp, wrist) * 1.15;
  const pinchDist = dist2D(thumbTip, indexTip);

  // Pinch: thumb and index very close, other fingers must be curled
  if (pinchDist < 0.04 && mC && rC && pC) return 'pinch';

  // Peace: only index + middle extended, ring + pinky clearly curled
  if (iE && mE && rC && pC) return 'peace';

  // Point: only index extended, all others clearly curled
  if (iE && mC && rC && pC) return 'point';

  // Fist: all fingers clearly curled
  if (iC && mC && rC && pC && !thE) return 'fist';

  // Thumbs up: all fingers curled, thumb extended and above index base
  if (iC && mC && rC && pC && thE && thumbTip.y < indexMcp.y - 0.08) return 'thumbsup';

  // Open hand: all 4 fingers + thumb extended
  if (iE && mE && rE && pE && thE) return 'open';

  return 'none';
}

// ═══════════════════════════════════════════════════════════════════
//  Hand Tracking
// ═══════════════════════════════════════════════════════════════════
const videoEl = document.getElementById('webcam');
const overlay = document.getElementById('hand-overlay');
const ctx = overlay.getContext('2d');
let handLandmarker = null;
let lastVidTime = -1;
const CONNS=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];

function drawHand(lm){
  ctx.clearRect(0,0,overlay.width,overlay.height);
  if(!lm)return;
  const w=overlay.width,h=overlay.height;
  ctx.strokeStyle='rgba(255,111,216,0.7)';ctx.lineWidth=2;
  for(const[a,b]of CONNS){ctx.beginPath();ctx.moveTo(lm[a].x*w,lm[a].y*h);ctx.lineTo(lm[b].x*w,lm[b].y*h);ctx.stroke();}
  ctx.fillStyle='white';
  for(const l of lm){ctx.beginPath();ctx.arc(l.x*w,l.y*h,3,0,Math.PI*2);ctx.fill();}
}

function onHand(results) {
  if (performance.now() < keyboardOverrideUntil) return;

  if (results.landmarks && results.landmarks.length > 0) {
    const lm = results.landmarks[0];
    drawHand(lm);

    handPos.x = (lm[9].x - 0.5) * 2;
    handPos.y = -(lm[9].y - 0.5) * 2;

    const raw = recognizeGesture(lm);

    // Only act when the same gesture is detected consistently (5 consecutive frames)
    if (raw === lastRawGesture) gestureHoldTime++;
    else { gestureHoldTime = 0; lastRawGesture = raw; return; }
    lastRawGesture = raw;

    // Ignore ambiguous/none — only act on confident, held gestures
    if (raw === 'none') return;
    if (gestureHoldTime < 5) return;

    const prev = gestureSmoothed;
    gestureSmoothed = raw;

    // Continuous gestures
    if (raw === 'open') doExpand(true);
    else if (raw === 'fist') doContract(true);
    else if (raw === 'point') doAttract(true, handPos.x, handPos.y);
    else { doExpand(false); doContract(false); doAttract(false); }

    // One-shot gestures — only fire on transition
    if (prev !== raw) {
      if (raw === 'peace') doNextTemplate();
      if (raw === 'thumbsup') doCycleColor();
      if (raw === 'pinch') doFirework();
    }
  } else {
    gestureSmoothed = 'none';
    gestureHoldTime = 0;
    lastRawGesture = 'none';
    doExpand(false); doContract(false); doAttract(false);
    ctx.clearRect(0,0,overlay.width,overlay.height);
  }
}

async function initHands() {
  document.getElementById('gesture-label').textContent = 'Loading hand model...';
  try {
    const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm');
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', delegate:'GPU' },
      runningMode:'VIDEO', numHands:1,
    });
    const stream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},facingMode:'user'}});
    videoEl.srcObject = stream;
    await new Promise(r=>{videoEl.onloadedmetadata=r;});
    await videoEl.play();
    overlay.width = videoEl.videoWidth; overlay.height = videoEl.videoHeight;
    document.getElementById('gesture-label').textContent = 'Show your hand...';
    (function detect(){
      if(videoEl.readyState>=2 && videoEl.currentTime!==lastVidTime){
        lastVidTime=videoEl.currentTime;
        try{onHand(handLandmarker.detectForVideo(videoEl,performance.now()));}catch(e){}
      }
      requestAnimationFrame(detect);
    })();
  } catch(e) {
    console.error('[Hands]', e);
    document.getElementById('gesture-label').textContent = 'No camera — use buttons';
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Keyboard
// ═══════════════════════════════════════════════════════════════════
const keysDown = new Set();

document.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  keysDown.add(k);
  keyboardOverrideUntil = performance.now() + 3000;

  if (k==='n'||k==='arrowright') doNextTemplate();
  if (k==='p'||k==='arrowleft') doPrevTemplate();
  if (k==='c') doCycleColor();
  if (k==='f') doFirework();
  if (k==='o') doExpand(true);
  if (k==='x') doContract(true);
  if (k==='a') doAttract(true, 0, 0);
  if (k==='r') doReset();
  if (k>='1'&&k<='9') doSetTemplate(parseInt(k)-1);
});

document.addEventListener('keyup', (e) => {
  const k = e.key.toLowerCase();
  keysDown.delete(k);
  if (k==='o') doExpand(false);
  if (k==='x') doContract(false);
  if (k==='a') doAttract(false);
});

// ═══════════════════════════════════════════════════════════════════
//  Resize
// ═══════════════════════════════════════════════════════════════════
window.addEventListener('resize', () => {
  cam.aspect = window.innerWidth/window.innerHeight;
  cam.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
});

// ═══════════════════════════════════════════════════════════════════
//  Boot
// ═══════════════════════════════════════════════════════════════════

initParticles();
animate();
initHands().then(() => {
  const ld = document.getElementById('loading');
  ld.classList.add('hidden');
  setTimeout(()=>ld.remove(), 500);
});

console.log('=== READY ===');
console.log('Buttons on left side of screen');
console.log('Keys: N=next, P=prev, C=color, F=firework, O(hold)=expand, X(hold)=contract, A(hold)=attract, R=reset');
