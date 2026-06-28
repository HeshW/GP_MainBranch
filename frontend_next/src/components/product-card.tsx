import Link from "next/link";
import Image from "next/image";
import type { Product } from "@/lib/catalog";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

type ProductCardProps = {
  product: Product;
  onAddToCart?: () => void;
};

export function ProductCard({ product, onAddToCart }: ProductCardProps) {
  return (
    <Card className="group flex h-full flex-col overflow-hidden p-0">
      <Link href={`/products/${product.id}`} className="block">
        <div className="relative m-3 aspect-[4/3] overflow-hidden rounded-2xl bg-[var(--brand-soft)]">
          <Image
            src={product.image}
            alt={product.title}
            fill
            sizes="(min-width: 1024px) 25vw, (min-width: 768px) 50vw, 100vw"
            className="object-cover transition duration-500 group-hover:scale-105"
          />
        </div>
      </Link>
      <div className="flex flex-1 flex-col gap-4 p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <Badge>{product.category}</Badge>
            <Link href={`/products/${product.id}`}>
              <h3 className="mt-3 text-lg font-semibold leading-7 text-[var(--brand-heading)] transition group-hover:text-[var(--brand-primary)]">
                {product.title}
              </h3>
            </Link>
          </div>
          <span className="text-lg font-semibold text-[var(--brand-heading)]">${product.price}</span>
        </div>
        <p className="line-clamp-3 text-sm leading-6 text-slate-600">{product.description}</p>
        <div className="mt-auto flex items-center justify-between gap-3 pt-3">
          <span className="text-sm font-medium text-slate-500">
            {product.rating.rate.toFixed(1)} rating · {product.rating.count} reviews
          </span>
          <Button type="button" onClick={onAddToCart} className="px-4 py-2 text-xs">
            Add to cart
          </Button>
        </div>
      </div>
    </Card>
  );
}
