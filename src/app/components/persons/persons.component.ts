import { Component, input, output } from '@angular/core';

import { FaceService } from '../../services/face.service';
import { ImageElementService } from '../../services/image-element.service';
import { FilePathService } from '../views/file-path.service';

import type { PersonRecord } from '../../../../node/face/face-types';

/**
 * Person list view (design doc §8.2): a grid of person cards with the
 * representative face, an editable name, video/face counts and -- when
 * the attributes feature is on -- the per-person aggregates (§5.8).
 * Merging / splitting / reassigning faces is Phase 3 and not offered here.
 */
@Component({
  standalone: false,
  selector: 'app-persons',
  templateUrl: './persons.component.html',
  styleUrls: ['./persons.component.scss']
})
export class PersonsComponent {

  readonly folderPath = input<string>();
  readonly hubName = input<string>();
  readonly darkMode = input<boolean>();
  readonly attributesEnabled = input<boolean>(false);

  readonly personClicked = output<string>();
  readonly scanRequested = output<void>();

  editingPersonId = '';
  editingName = '';

  constructor(
    public faceService: FaceService,
    public filePathService: FilePathService,
    public imageElementService: ImageElementService
  ) { }

  /** Visible persons first (most videos first), hidden ones last. */
  persons(): PersonRecord[] {
    return [...this.faceService.persons].sort((a, b) => {
      if (a.hidden !== b.hidden) {
        return a.hidden ? 1 : -1;
      }
      return (b.videoHashes.length - a.videoHashes.length) || (b.faceCount - a.faceCount);
    });
  }

  cropPath(person: PersonRecord): string {
    return this.filePathService.createFaceCropPath(this.folderPath(), this.hubName(), person.representativeFaceId);
  }

  startEdit(person: PersonRecord): void {
    this.editingPersonId = person.personId;
    this.editingName = person.name;
  }

  commitEdit(person: PersonRecord): void {
    if (this.editingPersonId !== person.personId) {
      return;
    }
    this.editingPersonId = '';
    if (this.editingName.trim() !== person.name) {
      this.faceService.renamePerson(person.personId, this.editingName);
    }
  }

  toggleHidden(person: PersonRecord): void {
    this.faceService.setPersonHidden(person.personId, !person.hidden);
  }

  recluster(): void {
    this.faceService.requestRecluster(this.imageElementService.imageElements);
  }

  recalcAttributes(): void {
    this.faceService.requestAttributes(this.imageElementService.imageElements);
  }

  /** "♂ · 30s · 93%" -- sex symbol, age band, majority agreement (§5.8). */
  attributeLabel(person: PersonRecord): string {
    if (!person.attributes) {
      return '';
    }
    const attr = person.attributes;
    const symbol = attr.sex === 'M' ? '♂' : '♀';
    const band = Math.floor(attr.ageMedian / 10) * 10;
    return symbol + ' · ' + band + 's · ' + Math.round(attr.sexAgreement * 100) + '%';
  }

}
