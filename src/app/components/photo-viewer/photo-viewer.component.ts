import { Component, HostListener, input, output } from '@angular/core';

import { FilePathService } from '../views/file-path.service';

import type { ImageElement } from '../../../../interfaces/final-object.interface';

/**
 * Built-in photo viewer (docs/PHOTO_SUPPORT_DESIGN.md Phase 3):
 * a minimal lightbox over the hub-side preview copy
 * (previews/<hash>.jpg, long edge 1920px). Clicking the image toggles
 * between fit-to-window and actual size; arrow keys and the on-screen
 * buttons step through the hub's photos; Esc or a backdrop click
 * closes. The original file can still be opened externally.
 */
@Component({
  standalone: false,
  selector: 'app-photo-viewer',
  templateUrl: './photo-viewer.component.html',
  styleUrls: ['./photo-viewer.component.scss']
})
export class PhotoViewerComponent {

  readonly item = input<ImageElement>();
  readonly folderPath = input<string>();
  readonly hubName = input<string>();
  readonly positionText = input<string>();

  readonly closed = output<void>();
  readonly navigate = output<number>();
  readonly openExternally = output<void>();

  zoomed = false;

  constructor(
    public filePathService: FilePathService,
  ) { }

  get previewPath(): string {
    const item = this.item();
    if (!item) {
      return '';
    }
    return this.filePathService.createFilePath(this.folderPath(), this.hubName(), 'previews', item.hash);
  }

  toggleZoom(event: MouseEvent): void {
    event.stopPropagation();
    this.zoomed = !this.zoomed;
  }

  onBackdropClick(): void {
    this.closed.emit();
  }

  onNavigate(delta: number, event: MouseEvent): void {
    event.stopPropagation();
    this.zoomed = false;
    this.navigate.emit(delta);
  }

  onOpenExternally(event: MouseEvent): void {
    event.stopPropagation();
    this.openExternally.emit();
  }

  @HostListener('window:keydown', ['$event'])
  onKeydown(event: KeyboardEvent): void {
    if (!this.item()) {
      return;
    }
    if (event.key === 'Escape') {
      this.closed.emit();
    } else if (event.key === 'ArrowLeft') {
      this.zoomed = false;
      this.navigate.emit(-1);
    } else if (event.key === 'ArrowRight') {
      this.zoomed = false;
      this.navigate.emit(1);
    }
  }

}
