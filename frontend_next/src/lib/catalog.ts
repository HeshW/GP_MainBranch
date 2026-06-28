export type Product = {
  id: number;
  title: string;
  price: number;
  category: string;
  description: string;
  image: string;
  rating: {
    rate: number;
    count: number;
  };
};

export const featuredProducts: Product[] = [
  {
    id: 1,
    title: "Aero Trail Runner",
    price: 129,
    category: "Footwear",
    description:
      "Lightweight runners built for all-day movement, city walks, and weekend escapes.",
    image:
      "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=900&q=80",
    rating: { rate: 4.8, count: 148 },
  },
  {
    id: 2,
    title: "Studio Leather Tote",
    price: 189,
    category: "Accessories",
    description:
      "A structured leather tote with a clean silhouette and enough room for daily essentials.",
    image:
      "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?auto=format&fit=crop&w=900&q=80",
    rating: { rate: 4.6, count: 92 },
  },
  {
    id: 3,
    title: "Signal Wireless Headset",
    price: 249,
    category: "Audio",
    description:
      "High-clarity wireless headset with deep bass, long battery life, and soft ear cushions.",
    image:
      "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=900&q=80",
    rating: { rate: 4.9, count: 211 },
  },
  {
    id: 4,
    title: "Nova Desk Lamp",
    price: 84,
    category: "Home",
    description:
      "A warm, minimal desk lamp that gives your workspace a calmer, more focused feel.",
    image:
      "https://images.unsplash.com/photo-1513506003901-1e6a229e2d15?auto=format&fit=crop&w=900&q=80",
    rating: { rate: 4.4, count: 67 },
  },
];

export const productCategories = ["All", "Footwear", "Accessories", "Audio", "Home", "Fashion"];

export const storeHighlights = [
  {
    title: "Fast shipping",
    description: "Fulfillment designed for quick turnaround and transparent order tracking.",
  },
  {
    title: "Flexible checkout",
    description: "Simple order flow with clear totals, shipping, and contact details.",
  },
  {
    title: "Reliable support",
    description: "A clean customer experience across browsing, cart, and account pages.",
  },
];

export function getProductById(id: number) {
  return featuredProducts.find((product) => product.id === id);
}
