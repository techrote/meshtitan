import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { GLTFLoader } from "./vendor/GLTFLoader.js";

export function createViewer(host, { thumbnail = false } = {}) {
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    preserveDrawingBuffer: thumbnail,
  });
  renderer.setPixelRatio(thumbnail ? 1 : Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.append(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minDistance = 0.3;
  controls.maxDistance = 20;
  controls.autoRotateSpeed = 1.2;
  scene.add(new THREE.HemisphereLight(0xdde9ff, 0x4b4943, 2.8));
  for (const [color, intensity, position] of [
    [0xffebd3, 3.5, [3, 5, 4]],
    [0xaed6e7, 2, [-4, 2, -2]],
  ]) {
    const light = new THREE.DirectionalLight(color, intensity);
    light.position.set(...position);
    scene.add(light);
  }
  const grid = new THREE.GridHelper(12, 24, 0x40505c, 0x2b3844);
  grid.position.y = -1;
  grid.material.transparent = true;
  grid.material.opacity = 0.45;
  scene.add(grid);
  let model,
    generation = 0;
  let wireframe = false;
  function dispose(object) {
    object.traverse((child) => {
      if (!child.isMesh) return;
      child.geometry.dispose();
      const materials = Array.isArray(child.material)
        ? child.material
        : [child.material];
      for (const material of materials) {
        for (const value of Object.values(material))
          if (value?.isTexture) value.dispose();
        material.dispose();
      }
    });
  }
  function fit() {
    controls.target.set(0, 0, 0);
    camera.position.set(3.3, 2.1, 3.7);
    controls.update();
  }
  fit();
  new ResizeObserver(() => {
    const width = host.clientWidth,
      height = host.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }).observe(host);
  if (!thumbnail)
    renderer.setAnimationLoop(() => {
      if (document.hidden || !host.getClientRects().length) return;
      controls.update();
      renderer.render(scene, camera);
    });
  return {
    fit,
    setColor(value) {
      model?.traverse((child) => {
        if (child.isMesh) child.material.color.set(value);
      });
    },
    async snapshot() {
      renderer.setSize(host.clientWidth || 320, host.clientHeight || 240);
      camera.aspect = (host.clientWidth || 320) / (host.clientHeight || 240);
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
      return new Promise((resolve, reject) =>
        renderer.domElement.toBlob((blob) => {
          if (blob) resolve(blob);
          else reject(new Error("Could not render thumbnail"));
        }, "image/png"),
      );
    },
    setWireframe(value) {
      wireframe = value;
      model?.traverse((child) => {
        if (child.isMesh) child.material.wireframe = value;
      });
    },
    setRotate(value) {
      controls.autoRotate = value;
    },
    async load(url, upAxis = "y") {
      const request = ++generation;
      const gltf = await new GLTFLoader().loadAsync(url);
      if (request !== generation) {
        dispose(gltf.scene);
        return false;
      }
      if (model) {
        scene.remove(model);
        dispose(model);
      }
      model = gltf.scene;
      // TripoSR meshes use Z-up; the viewer's camera and orbit controls use Y-up.
      if (upAxis === "z") model.rotateX(-Math.PI / 2);
      model.traverse((child) => {
        if (!child.isMesh) return;
        const originals = Array.isArray(child.material)
          ? child.material
          : [child.material];
        originals.forEach((material) => material.dispose());
        child.material = new THREE.MeshStandardMaterial({
          color: 0xd2d7d4,
          roughness: 0.7,
          metalness: 0.1,
          side: THREE.DoubleSide,
          wireframe,
        });
        if (!child.geometry.attributes.normal)
          child.geometry.computeVertexNormals();
      });
      const box = new THREE.Box3().setFromObject(model);
      const size = box.getSize(new THREE.Vector3());
      const longest = Math.max(size.x, size.y, size.z);
      if (!Number.isFinite(longest) || longest <= 0) {
        dispose(model);
        model = null;
        throw new Error("Mesh has no visible geometry.");
      }
      const center = box.getCenter(new THREE.Vector3());
      const scale = 2 / longest;
      model.scale.multiplyScalar(scale);
      model.position.sub(center.multiplyScalar(scale));
      grid.position.y = (-size.y * scale) / 2 - 0.015;
      scene.add(model);
      fit();
      return true;
    },
  };
}
