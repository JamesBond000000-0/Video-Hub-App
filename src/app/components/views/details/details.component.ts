import { Component, input, output, viewChild } from '@angular/core';

import { FilePathService } from '../file-path.service';
import { FaceService } from './../../../services/face.service';
import { ImageElementService } from './../../../services/image-element.service';
import { ManualTagsService } from '../../tags-manual/manual-tags.service';

import type { BehaviorSubject } from 'rxjs';
import type { ElementRef } from '@angular/core';
import type { StarRating, ImageElement } from '../../../../../interfaces/final-object.interface';
import type { VideoClickEmit, RightClickEmit, TagEmit, RenameFileResponse } from '../../../../../interfaces/shared-interfaces';

export interface YearEmission {
  index: number;
  year: number;
}

export interface FaceDisplayGroup {
  cropPath: string;   // file:// URL of the representative display crop
  timesLabel: string; // appearance times, e.g. "0:12, 1:03"
  personId: string;   // clustered person ('' when unclustered)
  personName: string; // user-assigned person name ('' when unnamed)
  personNumber: number; // numeric part of the person ID, for the fallback label
}

@Component({
  standalone: false,
  selector: 'app-details-item',
  templateUrl: './details.component.html',
  styleUrls: [
      '../time-and-rez.scss',
      './details.component.scss',
      '../selected.scss'
    ]
})
export class DetailsComponent {

  readonly filmstripHolder = viewChild<ElementRef>('filmstripHolder');

  readonly filterTag = output<TagEmit>();
  readonly filterPerson = output<string>();
  readonly videoClick = output<VideoClickEmit>();
  readonly rightClick = output<RightClickEmit>();
  readonly sheetClick = output<void>();
  readonly refreshPlaylist = output<void>();

  readonly video = input<ImageElement>();

  readonly settingsButtons = input<any>();

  readonly connected = input<boolean>();
  readonly darkMode = input<boolean>();
  readonly draggable = input<boolean>();
  readonly elHeight = input<number>();
  readonly elWidth = input<number>();
  readonly folderPath = input<string>();
  readonly hoverScrub = input<boolean>();
  readonly hubName = input<string>();
  readonly imgHeight = input<number>();
  readonly largerFont = input<boolean>();
  readonly maxWidth = input<number>();
  readonly returnToFirstScreenshot = input<boolean>();
  readonly selectedSourceFolder = input<string>();
  readonly showAutoFileTags = input<boolean>();
  readonly showAutoFolderTags = input<boolean>();
  readonly showFavorites = input<boolean>();
  readonly showManualTags = input<boolean>();
  readonly showMeta = input<boolean>();
  readonly showTwoColumns = input<boolean>();
  readonly showVideoNotes = input<boolean>();
  readonly star = input<StarRating>();
  readonly thumbAutoAdvance = input<boolean>();

  readonly renameResponse = input<BehaviorSubject<RenameFileResponse>>();

  starRatingHack: StarRating; // updates stars when heart toggled in app-thumbnail

  constructor(
    public faceService: FaceService,
    public filePathService: FilePathService,
    public imageElementService: ImageElementService,
    public manualTagsService: ManualTagsService
  ) { }

  /**
   * Face groups of this video for the face strip (empty when the video
   * has no face data -- the strip renders nothing in that case)
   */
  faceGroups(): FaceDisplayGroup[] {
    const video = this.video();
    if (!video || !video.hash) {
      return [];
    }
    return this.faceService.faceGroupsForVideo(video.hash).map((group) => {
      const person = group.personId ? this.faceService.personById(group.personId) : undefined;
      return {
        cropPath: this.filePathService.createFaceCropPath(this.folderPath(), this.hubName(), group.representativeFaceId),
        timesLabel: group.times.map((t) => this.formatFaceTime(t)).join(', '),
        personId: group.personId,
        personName: person ? person.name : '',
        personNumber: person ? this.faceService.personNumber(person) : 0,
      };
    });
  }

  private formatFaceTime(totalSeconds: number): string {
    const rounded = Math.round(totalSeconds);
    const minutes = Math.floor(rounded / 60);
    const seconds = rounded % 60;
    return minutes + ':' + String(seconds).padStart(2, '0');
  }

  toggleHeart(): void {
    const video = this.video();
    this.starRatingHack = video.stars;
  }

}
