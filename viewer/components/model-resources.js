// Dispose each owned resource once, including paint hidden by inspection materials.
export function disposeModelResources(root, retainedMaterials = []) {
  const geometries = new Set(), materials = new Set(retainedMaterials.filter(Boolean));
  const textures = new Set(), skeletons = new Set(), images = new Set();
  const addMaterials = (value) => {
    for (const material of Array.isArray(value) ? value : [value]) if (material) materials.add(material);
  };
  root?.traverse((object) => {
    if (object.geometry) geometries.add(object.geometry);
    if (object.skeleton) skeletons.add(object.skeleton);
    addMaterials(object.material);
    addMaterials(object.userData?.paint);
  });
  for (const material of materials) {
    for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
  }
  for (const texture of textures) {
    if (texture.image) images.add(texture.image);
    texture.dispose();
  }
  for (const image of images) image.close?.();
  for (const material of materials) material.dispose();
  for (const geometry of geometries) geometry.dispose();
  for (const skeleton of skeletons) skeleton.dispose();
}
