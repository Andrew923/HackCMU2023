CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230226000000_e20230226235959_p20230227021641_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-27T02:16:41.417Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-26T00:00:00.000Z   time_coverage_end         2023-02-26T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxh�N�  �          @�G�����@p��aG���\C�����@{?���A��C�                                    Bxh�]&  �          @ʏ\��
=@&ff=�Q�?Y��C�=��
=@�?�AP��C)                                    Bxh�k�  �          @�
=����@-p�>��
@@��C�q����@
=q?�Az�HCǮ                                    Bxh�zr  �          @�G�����@[�=L��>�G�C�����@:�H?�ffA�p�C�                                    Bxh��  �          @�������@p  >�=q@{C)����@E�@
=qA��C�                                    Bxh���  "          @�33��\)@������uCY���\)@mp�@(�A�(�C:�                                    Bxh��d  T          @�Q���G�@|(���\)��
C���G�@aG�?��
Ak�C��                                    Bxh��
  T          @����{@Z=q=���?Y��C���{@7�?�=qA�ffC8R                                    Bxh�ð  �          @�(�����@N{?5@�p�C�����@��@��A���CT{                                    Bxh��V  �          @��H��=q@`  ?#�
@��C�3��=q@+�@ffA��HCG�                                    Bxh���  �          @�\)���H@dz�?W
=@�  Cu����H@(Q�@#33A�C��                                    Bxh��  "          @�ff���@\(�?�ffA��C�q���@=q@*�HA��C�3                                    Bxh��H  �          @�  ���
@G�?G�@�p�C����
@�@G�A�Q�C�f                                    Bxh��  �          @���G�@_\)?L��@�\)C�)��G�@%�@�RA��RC�q                                    Bxh��  T          @�����@n�R?��RAG�CW
���@(�@L(�Aܣ�CxR                                    Bxh�*:  �          @ҏ\��p�@fff?
=@�
=CaH��p�@2�\@
=A��C                                    Bxh�8�  �          @θR���@fff?���A"{C����@ ��@4z�A�
=Ck�                                    Bxh�G�  "          @�\)���@h��?�  Az�C+����@'
=@.�RA�{C�q                                    Bxh�V,  �          @���Å@i��?�\)A-�C!H�Å@(�@B�\Aƣ�C@                                     Bxh�d�  "          @�z��ʏ\@|��?�(�A2ffC��ʏ\@(��@R�\A̸RC\)                                    Bxh�sx  
�          @����=q@n�R?���AJffCp���=q@��@Q�AծC��                                    Bxh��  T          A���@z=q@\)AxQ�C����@{@{�A߮C"�{                                    Bxh���  
(          A{���@�=q@
=A�p�C+����@.{@��A�Q�C�                                     Bxh��j  �          @�(���z�@���?�33A`Q�C��z�@%@p��A�\C�                                    Bxh��  
�          A�H���@��@�
A��CQ����@ff@��HA�Q�C �=                                    Bxh���  �          @�33��(�@y��@��A��C����(�@��@���A�=qC!z�                                    Bxh��\  �          @����  @�=q?�\)AZ�HC�3��  @#33@k�A�z�C��                                    Bxh��  
�          A (�����@�(�@�
Ao\)C������@   @x��A��C                                    Bxh��  �          @����
=@�G�@Q�A�ffC���
=@��@�33A�G�C ��                                    Bxh��N  "          A\)��ff@��H@Q�A���C����ff@�
@�(�A��C!�                                    Bxh��  �          A��Ӆ@��@.�RA���C�)�Ӆ@z�@���B  C �f                                    Bxh��  "          A����  @��\?p��@���C�{��  @A�@8��A�z�C�=                                    Bxh�#@  "          @����@p  ?��
Az�CǮ��@$z�@@��A�{C5�                                    Bxh�1�  T          @������H@x��?�ffAYG�Cz����H@�@a�AۅC                                    Bxh�@�  "          @�(�����@z=q@%�A�  C�����@33@��RB�HC!8R                                    Bxh�O2  �          @�ff���@333?u@��HC�����?�@  A���C$��                                    Bxh�]�  �          @����@'
=?�(�A?\)C����?У�@Q�A���C"O\                                    Bxh�l~  �          @�����?���?��@�ffCY����?�33?��HAt��C$޸                                    Bxh�{$  
�          @�\)��ff?�  ��  �!�C"(���ff?��?+�@�  C#B�                                    Bxh���  �          @�p����@   ?   @�33C 0����?�(�?���Ab=qC%.                                    Bxh��p  �          @�������?�(�?:�H@�
=C%�����?fff?�\)A\Q�C*�q                                    Bxh��  �          @Å����?�G�?J=q@�ffC'�q����?.{?���AH  C-h�                                    Bxh���  
�          @ȣ����
@N�R?��AC�
C����
@�@1G�A�ffC�=                                    Bxh��b  T          @�{���\@
=?�{A&�RC����\?�(�@
=qA�(�C%5�                                    Bxh��  
�          @�G����@i��>�(�@�p�C�f���@;�@{A�  C�{                                    Bxh��  
�          @�{��p�?�
=?�{A<��C)�=��p�>���?�\Aw
=C1
=                                    Bxh��T  "          @�  ��Q�?�{?�33A*�\C)}q��Q�>\?ǮAg�
C0aH                                    Bxh���  
�          @���
=?Tz�?�p�A8��C,���
==�Q�?�p�A^�RC3�                                    Bxh��  T          @�  ���R>u?���A�C1�����R�5?�Q�A|  C:��                                    Bxh�F  "          @�
=���R>#�
?�  A�
=C2n���R�=p�?˅Am�C;�                                    Bxh�*�  �          @�G���33�W
=?��RAdz�C6���33�p��?�
=A3�
C=�                                    Bxh�9�  �          @Å����k�?���AH(�C<��������H?+�@˅CA��                                    Bxh�H8  �          @У��Å����?���AaG�C@p��Å����?+�@�  CE�q                                    Bxh�V�  �          @��\���R���?��Au�CAO\���R���?.{@�p�CG�                                    Bxh�e�  �          @�p������?ٙ�A�(�CKk����,��>Ǯ@|��CP�3                                    Bxh�t*  �          @�=q��녿�33@�A�=qCB�=�����R?�ffA$��CJ�\                                    Bxh���  �          @��
��=q?�G�@{A��RC%s3��=q=u@9��Aә�C3h�                                    Bxh��v  
�          @�z����@�  @<��A£�C8R���?�(�@�=qB=qC�                                    Bxh��  �          @����G�@�
=@<��A�(�C
����G�@
�H@�p�B&p�C��                                    Bxh���  T          @����{@1�@]p�A�\C���{?(��@�z�B�HC-
                                    Bxh��h  �          @߮��p�@��@eA���C�f��p�>��
@��\B\)C0��                                    Bxh��  �          A����33@��R@c33A�\)C
s3��33@   @�=qB(�HC�f                                    Bxh�ڴ  T          A	p����\@��@eAƏ\C
���\@'�@�p�B&=qC��                                    Bxh��Z  T          A  ����@�@Q�A�C������@QG�@�{B({C��                                    Bxh��   �          A����ff@�{@dz�A�
=C���ff@Vff@�=qB-�Ch�                                    Bxh��  "          A���p�@�Q�@w
=A��
C
c���p�@'�@�\)B*�
C&f                                    Bxh�L  �          A���@أ�@u�A���C��@|(�@ָRB0�CG�                                    Bxh�#�  �          A��
=@�{@U�A��CQ���
=@z=q@ÅB(��Ch�                                    Bxh�2�  
�          A�����@�33@]p�A��RB������@~�R@ə�B1ffC�=                                    Bxh�A>            A�\��G�@Ӆ@q�A�33C����G�@u@�=qB1(�CO\                                    Bxh�O�  
�          A(����\@ҏ\@QG�A�{CB����\@�=q@��
B&G�C�H                                    Bxh�^�  �          @���c33@ƸR@
=A��
B�u��c33@�ff@�z�B,G�C&f                                    Bxh�m0  �          @��H����@�  ?���AG�B�k�����@|��@j=qB$B�z�                                    Bxh�{�  �          @x���7����H?�33Aߙ�CJ���7���\)?Tz�AYCU�                                    Bxh��|  �          @W
=������?�Q�B�CM�)������?fffA�CZ�                                    Bxh��"  T          @���@�׿s33@%B�\CE���@����?�\A��
CW!H                                    Bxh���  
�          @����N�R�&ff@-p�B�C?O\�N�R��z�@�A�p�CR�)                                    Bxh��n  
�          @�
=�R�\�Ǯ@&ffB��C:Ǯ�R�\�У�@�
A�\CNh�                                    Bxh��  �          @���`�׿��@��B
Q�C<�f�`�׿�Q�?���A�G�CM�                                    Bxh�Ӻ  �          @�z��Dzῃ�
@#33B��CFxR�Dz��	��?ٙ�A�33CW�                                    Bxh��`  
�          @�p��AG�����@\)BQ�CK���AG��Q�?�  A�
=CZJ=                                    Bxh��  
�          @�G��=p���{@(�B�\CT!H�=p��5�?���A�(�C_��                                    Bxh���            @�33��z�@K�@G�A�p�C�R��z�?��H@�G�B!�C&{                                    Bxh�R  �          @�33��=q@.�R@QG�A��\C&f��=q?@  @�ffB(��C*��                                    Bxh��  T          @�(�����?�@�p�B$
=C%�H���Ϳ�ff@�ffB%��C@�3                                    Bxh�+�  
�          @y���6ff?n{@{BC!��6ff���@&ffB(p�C<:�                                    Bxh�:D  "          @XQ��7�?��?�
=A�
=C���7�?
=q?�A�=qC)Y�                                    Bxh�H�  T          @�G��Dz�?u@=qBz�C"���DzᾸQ�@$z�B  C:�f                                    Bxh�W�  "          @�\)���
?�\)@�  B\)C$O\���
�J=q@���B ffC=(�                                    Bxh�f6  T          @�=q��p�?�z�@r�\B�C$�R��p��#�
@�  B�C;�                                    Bxh�t�  
�          @����ff?�G�@c33B��C%��ff�(��@mp�B(�C;��                                    Bxh���  "          @�����?�33?uA'\)C!}q���?u?��A�\)C(�                                    Bxh��(  
�          @�ff����?z�H?�  A�\)C'�R����    @   A�33C4�                                    Bxh���  T          @����}p�?
=q@�\A�z�C,0��}p���@33A���C;��                                    Bxh��t  T          @��H�vff?���@&ffB\)C"���vff�k�@7
=B(�C7p�                                    Bxh��  �          @�ff��G�?���@*=qA�z�C$T{��G����@:=qB=qC7n                                    Bxh���  T          @�  ���?}p�@*�HA��HC'^������ff@3�
B�C9�
                                    Bxh��f  
�          @������?�\@
=A�C-Y�����5@�
A�z�C=�                                    Bxh��  
�          @��H��p�?5@{A��C+Y���p��
=@   A�(�C;=q                                    Bxh���  
�          @�33��  ?k�@��A�Q�C(ff��  ��=q@�A݅C7n                                    Bxh�X  |          @�����H?�\)?��A�{C#ff���H>��
@�RA�
=C0\                                    Bxh��  
�          @�  ���\?�{?�  A��C�����\?E�@�A��C*}q                                    Bxh�$�  �          @����p�@��@�A�(�C:���p�?n{@N�RA�p�C*J=                                    Bxh�3J  T          @�Q�����@*=q@X��AָRC������?(��@�Q�B
{C-ٚ                                    Bxh�A�  T          @�{���@:�H@L(�A��HCJ=���?}p�@�ffB	��C*�3                                    Bxh�P�  ^          @�Q��ʏ\@W
=@<(�A�z�C�ʏ\?�  @�ffB�\C&�f                                    Bxh�_<  |          @�\)���@^{@)��A�=qC޸���?޸R@�  BC##�                                    Bxh�m�  T          @�z���
=@���<#�
=L��Cٚ��
=@��@G�A��C��                                    Bxh�|�  T          @������@��?\(�@�p�C������@Vff@6ffA�CW
                                    Bxh��.  �          A��ff@�
=?ٙ�A@(�C����ff@S33@q�A�ffC�f                                    Bxh���  
�          Az���p�@���?n{@�
=C@ ��p�@o\)@H��A��HC��                                    Bxh��z  T          @�{��Q�@�=q?���A\)Cn��Q�@O\)@C33A�G�C�H                                    Bxh��   �          @�  �θR@�G�?�
=A*�\C��θR@5@Mp�AîCL�                                    Bxh���  �          A��׮@��?��HA&{C�\�׮@@��@U�A�Q�C�R                                    Bxh��l  �          A�\����@���?�ffA33C������@@  @I��A�Cz�                                    Bxh��  �          A�H���H@�\)?�\)@��CQ����H@fff@QG�A�G�C:�                                    Bxh��  T          A Q�����@��?��\@���Cp�����@W�@AG�A���C�                                    Bxh� ^  T          A(����
@�  ?u@�{C���
@^�R@AG�A��C&f                                    Bxh�  �          A�����@��H?���@�RC޸����@_\)@J�HA�(�C��                                    Bxh��  �          A����@��?k�@\Cٚ���@s�
@I��A���C��                                    Bxh�,P  �          A�R��@��R?�{@�{C��@���@^�RA���Cp�                                    Bxh�:�  �          AG���@�
=?c�
@�G�C���@{�@J�HA�z�C�                                     Bxh�I�  �          A
�R��p�@�=q?#�
@�Q�C)��p�@z=q@8Q�A�G�C^�                                    Bxh�XB  �          A
�H����@��H?=p�@�z�C�f����@y��@>{A�ffCn                                    Bxh�f�  �          A���Q�@���?�R@��HC�\��Q�@z=q@6ffA�Q�C��                                    Bxh�u�  �          A	���{@�\)?\)@p  C����{@xQ�@0��A�\)C�f                                    Bxh��4  �          A	����(�@���>�G�@>{C���(�@\)@+�A�\)C�                                    Bxh���  �          AQ����H@���>�=q?�{Ck����H@��
@4z�A�C�                                     Bxh���  �          AQ���p�@�(�>��@p�C�q��p�@�  @C33A�G�C@                                     Bxh��&  �          A����ff@��
?�@FffC����ff@�{@HQ�A�\)C�                                    Bxh���  �          A�����@��>�@0��CT{���@�p�@C33A�\)C�                                    Bxh��r  �          A  ���@�=q?L��@��HC� ���@���@O\)A�C33                                    Bxh��  �          Az���z�@�ff?\(�@��\CǮ��z�@��@N�RA�
=Cp�                                    Bxh��  �          A�����@�?��\@�{C����@�=q@W
=A��C.                                    Bxh��d  �          A����H@���?z�H@��C����H@�{@XQ�A�ffC
                                    Bxh�
  �          Aff��@��R?J=q@��\C^���@�@R�\A��C��                                    Bxh��  �          A�\����@��?J=q@���C������@�z�@P��A�=qC&f                                    Bxh�%V  "          A��Q�@���?\(�@�  Cp���Q�@�(�@L(�A�C��                                    Bxh�3�  �          A����\@���?���@�(�C&f��\@��\@c33A�\)C��                                    Bxh�B�  T          A�� ��@��?���@��
C�)� ��@xQ�@e�A�z�CJ=                                    Bxh�QH  �          A���(�@��
?�=q@���C8R�(�@xQ�@e�A���C��                                    Bxh�_�  T          A"{�Q�@�=q?�  @��
C�Q�@�33@g
=A�p�CO\                                    Bxh�n�  T          A"�\��@��
?��A\)C���@�Q�@x��A�  C��                                    Bxh�}:  T          A%p��
{@�?�(�A�C�
�
{@~�R@�=qA��C@                                     Bxh���  �          A&�\�
�R@���?У�A�
C}q�
�R@��@���A�z�C�H                                    Bxh���  �          A%��	�@�p�?�@�(�C}q�	�@��H@z�HA��C#�                                    Bxh��,  �          A%G��	��@�(�?�  @���C�=�	��@�  @aG�A���C^�                                    Bxh���  T          A�H�  @a�@8Q�A�ffC���  ?�\@�ffA�C'�f                                    Bxh��x  �          A33��
@p  @1�A���C�=��
@ ��@��RA�p�C&O\                                    Bxh��  T          A�\��@o\)@J=qA��RC����?�=q@�G�A�z�C'��                                    Bxh���  �          Az��p�@~�R@K�A�  C���p�@�@�p�A�z�C&T{                                    Bxh��j  �          A=q�@�ff@A�A��\C���@33@�z�A�ffC$(�                                    Bxh�  T          A�\����@|��@QG�A�(�C!H����?�(�@��A�{C%�{                                    Bxh��  �          A�����@J=q@3�
A�33C�\����?��R@}p�A�\)C'c�                                    Bxh�\  �          @�33��ff?n{@XQ�B��C)�{��ff�@  @[�B�\C<5�                                    Bxh�-  T          @�z����@2�\?��A��C�����?��H@0  A�C��                                    Bxh�;�  �          @�
=����@'
=@�A�33C�����?�=q@EA��RC$��                                    Bxh�JN  �          @�{����@9��?У�A�{C^�����?�@2�\A��C��                                    Bxh�X�  �          @���ff@ ��@A��HC(���ff?�  @@��A�(�C%��                                    Bxh�g�  T          @�p�����@�@ffA���C�����?5@AG�A���C+�                                    Bxh�v@  �          @���\)@�@z�A�
=C\)��\)?\(�@Dz�A�  C*8R                                    Bxh���  �          @�������@�@�RA���CǮ����?s33@R�\B��C(�=                                    Bxh���  �          @�{���
@@��A�33CT{���
?s33@P��B{C)                                    Bxh��2  "          @�����G�@�R@'�A�
=C�q��G�?E�@VffB
�C*��                                    Bxh���  �          @��H��{@��@��A���C  ��{?p��@=p�A�(�C)8R                                    Bxh��~  �          @�z�����@(�@(��A�\)Cc�����?:�H@VffB
�C+Q�                                    Bxh��$  �          @�p����H@	��@'�Aң�C����H?5@Tz�B�\C+��                                    Bxh���  �          @�{���H@@{A�p�C�����H?W
=@;�A��C*�)                                    Bxh��p  T          @�=q���?�@�AƸRC$T{���>��@3�
A�z�C2W
                                    Bxh��  
�          @�����
=?޸R@
=qA��C!�{��
=?��@.{A�
=C-�                                    Bxh��  �          @\��  @�
@\)A���C����  ?O\)@<(�A癚C+E                                    Bxh�b  �          @�����Q�@�@��A�  Cs3��Q�?^�R@6ffA�\)C*��                                    Bxh�&  �          @�
=��=q?��?�
=A�33C ����=q?J=q@%�A̸RC+��                                    Bxh�4�  �          @�{��Q�?��H@��A�33C"��Q�?��@+�A�33C.
                                    Bxh�CT  �          @�\)��
=?���@W
=B	p�C$���
=����@fffB{C7�R                                    Bxh�Q�  �          @������?@  @�z�B;{C)�=��녿�=q@�  B3z�CF!H                                    Bxh�`�  �          @�\)��Q�?���@\��B  C'G���Q�(�@c�
BQ�C;J=                                    Bxh�oF  �          @����Q�?�\)@#33A��C$� ��Q�=u@8��A��C3L�                                    Bxh�}�  �          @������
?�=q?���A��C#����
>�Q�@�RA�33C/�                                    Bxh���  �          @�z����R?�\)?��RAU�C!�����R?aG�?�A�=qC)�3                                    Bxh��8  
�          @�����{?�
=��G���
=C�
��{?��
?E�A   C!�                                    Bxh���  �          @��R���\?�p��h���p�C!8R���\?��������RC                                      Bxh���  T          @��H����?�\)?�Al��C'�3����>�Q�?��
A��RC0\                                    Bxh��*  
�          @������R?��\?�\)A�(�C(�R���R>.{?�z�A���C2{                                    Bxh���  �          @�������?��
?У�A�
=C(z�����>8Q�?�A��HC1�                                    Bxh��v  T          @�Q�����?h��?��HA���C)�q����=#�
?�Q�A���C3�                                     Bxh��  T          @�  ���?s33?�G�A�33C)�f���=u@   A��C3O\                                    Bxh��  �          @����\)?���@�A�z�C(����\)���
@�HA���C48R                                    Bxh�h  T          @Å��33?�G�@z�A�z�C'G���33>W
=@�HA�z�C1�f                                    Bxh�  T          @�����z�?�G�?�ffAn=qC%���z�?(��@�
A��C-^�                                    Bxh�-�  �          @�G����R?�
=?���A#\)C#�H���R?��\?��HA�G�C)�
                                    Bxh�<Z  T          @�=q���@   ?Y��A ��C ����?�z�?�33A|��C%��                                    Bxh�K   T          @�ff����@+�>�{@Tz�C{����@  ?�  Aj�\C�                                    Bxh�Y�  T          @�{��G�@,(�>�  @��C\��G�@33?�A\��C��                                    Bxh�hL  
�          @�\)��
=@8Q�#�
���RC
��
=@)��?��A0z�C\                                    Bxh�v�  �          @������H@p�?!G�@�=qCO\���H?�?�33A���C 33                                    Bxh���  "          @��H��z�@��>.{?У�C�{��z�?��?�\)A1��C �H                                    Bxh��>  T          @�����\)@�>�z�@7
=C���\)@33?�=qAUp�C��                                    Bxh���  "          @�33���H@ff�k��33CG����H@p�?Tz�A=qC��                                    Bxh���  �          @������@Q�?��@��C������?�?�Q�Ai�C"n                                    Bxh��0  
�          @�����R?���?��
A((�Cz����R?�ff?��
A�33C&�                                    Bxh���  �          @�(���
=?�?Q�A�RCٚ��
=?�{?�=qA��RC%k�                                    Bxh��|  T          @��\��?�?��
A)��C ����?�
=?�(�A���C'+�                                    Bxh��"  T          @�G���p�?�p�?�G�A(  C!����p�?�{?�A�  C'�H                                    Bxh���  �          @�{��  ?���?�A=C �=��  ?�33?�{A���C'��                                    Bxh�	n  �          @������\?��?��AN{C �����\?���?��RA�ffC(                                    Bxh�  �          @�G���z�?�G�?�z�A8��C!����z�?���?�A��C(�q                                    Bxh�&�  �          @�ff����?�  ?���A8  C!�q����?�=q?��
A�{C(��                                    Bxh�5`  T          @�����?�\?�G�A$Q�C!������?�33?�
=A�C'�                                    Bxh�D  	�          @�����?��
?E�@�(�C!�)��?��\?��HAk33C&ٚ                                    Bxh�R�  T          @�
=����?�{?333@�ffC&�3����?n{?�(�A<��C*��                                    Bxh�aR  �          @����G�?�{?0��@��HC&����G�?n{?��HA:=qC*�                                    Bxh�o�  T          @�����H?�p�?(��@�  C%�=���H?�ff?��RA<(�C)�
                                    Bxh�~�  T          @Å���
?���?fffA�HC&#����
?k�?�
=AYG�C+�                                    Bxh��D  �          @�����=q?��?�R@�33C%8R��=q?�\)?�(�A:{C)�                                    Bxh���  �          @��
��(�?�ff?333@�33C%:���(�?��?�ffAD��C)s3                                    Bxh���  �          @�z����R?�{?�R@�  C'+����R?u?��A+33C*�)                                    Bxh��6  �          @�Q����?�\)?:�H@�
=C&�R���?k�?�  A?�C*�R                                    Bxh���  	�          @�
=��Q�?��?5@�=qC&n��Q�?s33?��RA?�C*��                                    Bxh�ւ  �          @�(���
=?��
?\)@���C'���
=?h��?��A�RC+T{                                    Bxh��(  T          @�ff��G�?�p�?(��@���C(p���G�?Tz�?���A(  C,(�                                    Bxh���  T          @�Q���G�?��R?5@�\)C&)��G�?�ff?��
A<Q�C*(�                                    Bxh�t  �          @�������?��H?^�R@�\)C&ff����?s33?�z�AP  C+�                                    Bxh�  �          @�  ��Q�?�\)?�G�A��C'#���Q�?O\)?�  A^{C,T{                                    Bxh��  
Z          @Ǯ��=q?�ff?�@�  C'����=q?n{?���A=qC+B�                                    Bxh�.f  
�          @�����
?�\)?(��@�G�C'\)���
?u?�
=A-G�C+
                                    Bxh�=  �          @�G����H?���?.{@�p�C'=q���H?u?���A/�C+�                                    Bxh�K�  T          @\����?�>�
=@|��C&s3����?�{?z�HA��C)W
                                    Bxh�ZX  T          @�����\?���>��
@A�C&�����\?��?\(�A�
C)k�                                    Bxh�h�  T          @�p���Q�?��>Ǯ@r�\C'33��Q�?��\?h��A�C)��                                    Bxh�w�  
�          @�ff����?�ff>�G�@�ffC'aH����?}p�?p��AC*J=                                    Bxh��J  �          @�����(�?�>\)?���C(�q��(�?�  ?�R@�ffC*W
                                    Bxh���  T          @�����?��H?��A$��C"����?���?�
=A��C(�=                                    Bxh���  �          @����33?���?�(�As�C O\��33?��@ffA�(�C(�                                    Bxh��<  �          @��\���H?�{?�=qA[�
C�����H?���?��RA��RC'n                                    Bxh���  �          @�(���p�?�33?�A?�
C�\��p�?��R?�{A�=qC&�\                                    Bxh�ψ  
�          @��H���?�{?��
AS�C�����?�z�?�Q�A��\C'@                                     Bxh��.  "          @��
���R?ٙ�?��RAK33C!�3���R?��
?�=qA�=qC(ٚ                                    Bxh���  �          @��R���
?�=q?���A1��C#�)���
?z�H?�z�A��C)��                                    Bxh��z  
�          @�33���\?��H?c�
AC$�����\?u?�AjffC)Ǯ                                    Bxh�
   �          @�(����H?Ǯ?^�RA�\C#�R���H?��?�Q�Amp�C(��                                    Bxh��  T          @�����?�Q�?���A'\)C#����?���?�
=A�{C(ٚ                                    Bxh�'l  "          @�(�����?��
?�Q�ADQ�C#�{����?fff?�(�A��RC*Y�                                    Bxh�6  �          @�Q���(�?��
?��RAO�
C#\)��(�?aG�?�G�A�\)C*:�                                    Bxh�D�  
�          @�����?�(�?���A>�RC$!H���?^�R?��A�
=C*s3                                    Bxh�S^  �          @�=q��G�?��?�=qAhz�C����G�?�?��RA�{C&T{                                    Bxh�b  _          @����?�
=?�(�Az�HCaH���?�33@Q�A���C&��                                    Bxh�p�  I          @�=q��=q?��
?��RAw�
C ����=q?�G�@A�{C(�q                                    Bxh�P  �          @�Q�����?�G�?�Ao\)C ������?��\@G�A�=qC(z�                                    Bxh���  �          @��R��?�G�?�\)A�(�C!5���?p��@��A��C)�3                                    Bxh���  �          @�33��G�?��?�  A��HC!T{��G�?k�@�A�G�C*.                                    Bxh��B  T          @����z�?���?�G�A��HC =q��z�?xQ�@
=A�Q�C)\)                                    Bxh���  �          @�G����?�ff?˅A��C 5����?}p�@(�A��C(�                                    Bxh�Ȏ  �          @�=q��ff?�\?�A�=qC Q���ff?^�R@��A�=qC*�                                    Bxh��4  �          @�G���?�\)?��A5�C"����?��
?�z�A�33C(��                                    Bxh���  �          @�=q��
=?��?���A\  C%&f��
=?:�H?�\A�z�C,\                                    Bxh��  �          @�{����?�33?��HAx��C$xR����?0��?�33A�C,5�                                    Bxh�&  �          @�=q��(�?�?�33A�p�C$����(�?!G�@�A�G�C,��                                    Bxh��  �          @�Q����\?��?˅A��
C$  ���\?\)?��RA�ffC-\                                    Bxh� r  
�          @����p�?��?�A�
=C$�R��p�>��?�G�A��C-�\                                    Bxh�/  T          @�(����?��?��A��\C&aH���>�
=?�33A���C.�                                     Bxh�=�  �          @�Q���  ?+�?��A���C+
=��  =�\)?�  A��C3                                    Bxh�Ld  
Z          @��H��33?(�?�G�A�p�C+�\��33=L��?�33A���C3L�                                    Bxh�[
  
(          @�����
?333?�z�A�
=C*G����
=��
?���A�33C2޸                                    Bxh�i�  �          @���p�׼�?\A�ffC4u��p�׿#�
?���A��C=��                                    Bxh�xV  
(          @j�H�Y���8Q�?��A�\)C6���Y���333?�33A��C?�\                                    Bxh���  "          @��\���<#�
?�z�A�C3�����\)?��Ax��C:�H                                    Bxh���  �          @p��Q켣�
?�p�A�C4z��Q�   ?�\)AظRCA\)                                    Bxh��H  
�          @w
=�^{>#�
?˅A�p�C1Q��^{���H?��
A���C;�R                                    Bxh���  �          @�{��(�>��?�Q�A��C.���(��u?�p�A�Q�C6��                                    Bxh���  �          @����G�>�=q?��HA�(�C0.��G�����?�Q�A���C8�f                                    Bxh��:  
Z          @�=q���>���?��
A�z�C/�H��녾��R?��A���C7�
                                    Bxh���  T          @�  ��\)>��?�z�A��HC.n��\)�8Q�?��HA�{C6h�                                    Bxh��  
�          @�\)��ff?   ?\A�{C-����ff�\)?˅A��C5�)                                    Bxh��,  �          @p���dz�>Ǯ?���A�
=C-���dz὏\)?��A�p�C5�                                    Bxh�
�  
�          @�G���?���?��
A���C'+���>\?���A���C/W
                                    Bxh�x  
�          @����{�?�G�?�=qA�\)C%�H�{�>���?�\)A�Q�C.#�                                    Bxh�(  �          @�\)�u?��
?�
=A��C%  �u>Ǯ?�(�A�p�C.8R                                    Bxh�6�  T          @}p��`  ?���?��A�\)C!��`  ?z�?��HA�33C*�{                                    Bxh�Ej  
�          @w��\(�?���?�z�A�G�C"�=�\(�>�(�?�(�AҸRC,�H                                    Bxh�T  �          @�  �\��?�  ?��HA��HC��\��?p��@��B
=C$�3                                    Bxh�b�  �          @���mp�?�\)?ǮA���C���mp�?(��?�(�A�\)C)�R                                    Bxh�q\  
�          @�\)�mp�?�
=?�{A�\)C"L��mp�>��?�Q�A�\)C,��                                    Bxh��  "          @j�H�@  ?xQ�?��
A���C"�@  >W
=@G�B(�C0                                      Bxh���  "          @X���7�?u?\A�
=C!�
�7�>�z�?�\A�{C.&f                                    Bxh��N  
�          @K��2�\?(��?���A�=qC&�=�2�\=��
?�  A�=qC2T{                                    Bxh���  
�          @Z�H�Dz�?O\)?�  A��\C%@ �Dz�>��?�(�A�  C/!H                                    Bxh���  T          @H���(��?Y��?�Q�A�ffC"��(��>k�?�z�B �\C/                                      Bxh��@  "          @g
=�8Q�?�?�ffA��C�{�8Q�>Ǯ@
=B�C,B�                                    Bxh���  
�          @\���:�H?c�
?\A�G�C#)�:�H>u?޸RA��C/Y�                                    Bxh��  	�          @6ff�   ?0��?��A��C$���   >L��?���A��C/^�                                    Bxh��2  �          @n�R�G�?n{?�ffA�\)C#k��G�>.{@G�B\)C0޸                                    Bxh��  
�          @fff�@  ?G�?���A��C%� �@  <��
?�p�B�
C3��                                    Bxh�~  
Z          @n{�C�
?aG�?��A�{C#���C�
=�Q�@B	=qC2=q                                    Bxh�!$  _          @aG��7�?fff?�=qA�z�C"���7�>\)@�\B=qC1@                                     Bxh�/�  {          @U��333?G�?�\)A��HC$xR�333=�G�?��Bp�C1�{                                    Bxh�>p  "          @c�
�L(�?
=?�Q�A�(�C)���L(�    ?ǮA�z�C4�                                    Bxh�M  "          @���aG�?��@ffBffC"��aG�=�@%B�C2\                                    Bxh�[�  
�          @�Q��[�?�=q@�\B�
C"�
�[�>�@!G�BG�C1�=                                    Bxh�jb  "          @u��P  ?n{?��
A�ffC#�R�P  >B�\@   A��HC0��                                    Bxh�y  "          @�
=�{�?n{?��A��
C&�3�{�>��@A�{C1�                                    Bxh���  T          @�ff�\)?5?�\A���C)�)�\)    ?�z�A̸RC4\                                    Bxh��T  �          @��\�y��>��
?�A��C/O\�y���\)?���A��C6{                                    Bxh���  
Z          @�\)��
=>��
?��A�{C0&f��
=��\)?�ffA��RC7ff                                    Bxh���  �          @�����  >�?�{Ai��C233��  ��z�?�=qAc33C7�f                                    Bxh��F  
�          @Mp��H�ý�Q�?#�
A7\)C5�q�H�þ���?��A�C9�                                    Bxh���  
�          @���z�=#�
?�\@�\C3ff��z���>��H@�G�C6
                                    Bxh�ߒ  �          @�\)���
>8Q�?
=@�ffC1�����
�L��?�RA z�C4�f                                    Bxh��8  �          @��׿�\)@"�\@}p�BR��B����\)?fff@��
B�8RC�                                    Bxh���  �          @��R��=q@��@���B^��B��R��=q?333@�(�B���CL�                                    Bxh��  �          @�=q��
=@5@���Ba�B�
=��
=?n{@��
B���C�
                                    Bxh�*  �          @��Ϳ�=q@z�@���Bu\)B�=q��=q>�33@��B���C"�                                    Bxh�(�  �          @�33�Fff?�Q�@ffA���C)�Fff?\)@\)B{C)�3                                    Bxh�7v  
Z          @e�J�H?�=q?�G�A�\)CQ��J�H?\(�?�
=A�  C$��                                    Bxh�F  
�          @�{�n{?�\)?�(�A���C���n{?^�R@(�A�\C&Ǯ                                    Bxh�T�  
Z          @�  �n{?�G�?���A�Q�C���n{?z�H@�A��C%O\                                    Bxh�ch  �          @z=q�P  ?��
?�G�A��\C�=�P  ?^�R?�(�A�Q�C$�R                                    Bxh�r  "          @5��\)?z�H?�33A��CO\�\)>��?�z�BG�C)��                                    Bxh�  �          @8�ÿ�(�?��?�G�B�
C�R��(�>�33@�B5p�C*                                    BxhZ  T          @�(��B�\?�p�@��BG�C{�B�\?�\@1G�B'�C*��                                    Bxh   T          @����N�R?��\@	��B\)C"p��N�R>8Q�@�B�C0��                                    Bxh¬�  
�          @e�5�?�G�?���B  C W
�5�>u@(�BG�C/8R                                    Bxh»L  _          @s33�W�>\?�
=A���C-���W����?ٙ�A��
C8Y�                                    Bxh���  {          @��\�J�H?��@AG�B33CT{�J�H?+�@_\)B<��C(�                                    Bxh�ؘ  "          @��tz�?��H@AG�B
=C�)�tz�?�\@[�B&��C,Y�                                    Bxh��>  
�          @����Q�?�
=@2�\A�Q�C!k���Q�>�{@G�BQ�C/xR                                    Bxh���  �          @������H?�ff@,(�A�{C!W
���H>��H@C�
BG�C-�R                                    Bxh��  T          @�=q���
?��H@7�A�C�f���
?�@R�\B{C,�{                                    Bxh�0  T          @�{�}p�@{@Mp�B
{C��}p�?�z�@xQ�B,��C#�f                                    Bxh�!�  �          @\�r�\@,(�@z=qB =qC���r�\?�\)@��BE�\C#�                                    Bxh�0|  "          @����e�@�
@j�HB#�C=q�e�?Y��@�  BD�C&�                                    Bxh�?"  �          @����e�@-p�@��B(ffC�H�e�?��@�  BO(�C#�                                    Bxh�M�  T          @���w�@=q@fffB=qC��w�?z�H@�\)B:p�C%Ǯ                                    Bxh�\n  �          @�(����\@%�@qG�B��C�����\?�=q@�{B9��C%(�                                    Bxh�k  
�          @��R����@{@eBz�C�H����?�ff@��B5Q�C%�=                                    Bxh�y�  
�          @�G���ff@
=@i��B�\C����ff?n{@�  B2��C'�=                                    BxhÈ`  "          @�����\@z�@s�
B\)C^����\?Tz�@�(�B9�C(u�                                    Bxh×  T          @�33�|��@�
@���B%Q�C��|��?@  @��\BB�
C)8R                                    Bxhå�  
(          @Å��=q@��@uBC���=q?s33@��RB;G�C&�\                                    BxhôR  
�          @���vff@333@p  B�HC��vff?���@��B?\)C!)                                    Bxh���  �          @���w
=@*�H@fffB
=CG��w
=?�  @��B:��C"�                                    Bxh�ў  
Z          @�  �y��@&ff@p  B�RCJ=�y��?���@�p�B=��C#��                                    Bxh��D  �          @�{�qG�@(��@p  B��C��qG�?�
=@�BAG�C"�f                                    Bxh���  T          @��\�H��@2�\@l��B%\)C
c��H��?��@�{BQ��C�3                                    Bxh���  	�          @���_\)@333@j=qB{CG��_\)?�{@���BF\)C�3                                    Bxh�6  �          @Å��33@��@dz�B��C=q��33?�G�@�B,Q�C&��                                    Bxh��  
�          @�����Q�@ ��@XQ�BC�3��Q�?8Q�@w�B'ffC*Y�                                    Bxh�)�  T          @�������?�  @Dz�B��C�{����?
=@_\)B��C,��                                    Bxh�8(  �          @�ff���\?�33@9��A���C !H���\?\)@S33B33C-\                                    Bxh�F�  
�          @ƸR��  @   @J=qA�C33��  ?J=q@j=qBp�C*�R                                    Bxh�Ut  
�          @�Q���33@z�@Y��A�{C���33?��\@\)B��C)33                                    Bxh�d  
�          @�Q����@p�@c�
A���C}q���?\(�@�33B�C+E                                    Bxh�r�  
�          @�z����?���@AG�A��
C"�����?5@^{A�C-(�                                    Bxhāf  �          @�Q���z�@(�@G
=Aң�C�{��z�?�G�@j�HA�z�C*O\                                    BxhĐ  �          @أ���\)@��@P  A�{C�
��\)?��\@tz�B	�HC)k�                                    BxhĞ�  "          @Ӆ����@"�\@[�A�ffC5�����?�p�@��\B
=C&@                                     BxhĭX  T          @ҏ\���?���@U�A�{C�����?5@r�\B��C,:�                                    BxhĻ�  "          @�����Q�?�ff@XQ�A�  C!)��Q�?\)@r�\BG�C-�3                                    Bxh�ʤ  T          @�=q����?�Q�@4z�A�(�C'�f����>.{@C�
A�\)C2E                                    Bxh��J  �          @�����?�(�@(Q�Aљ�C!E���?:�H@C33A��RC+�                                     Bxh���  �          @����=q@,(�@,(�A���C\��=q?У�@[�B=qC"&f                                    Bxh���  T          @�  ��{@S�
@$z�A��C(���{@��@a�B�Cff                                    Bxh�<  
Z          @��H���R@#33@L��A���C8R���R?��@w
=B�RC#G�                                    Bxh��  �          @θR��33@Q�@u�B��C�{��33?xQ�@��B,��C(�                                    Bxh�"�  T          @θR����@!�@u�B�C�)����?�{@�ffB.��C&:�                                    Bxh�1.  "          @�z��u@�
@W�B��C��u?��@|��B2�
C$�)                                    Bxh�?�  T          @��R�j�H@��@j=qBffC�3�j�H?��@��B@
=C$!H                                    Bxh�Nz  �          @�(�����@��@W�B��C������?u@z=qB&G�C'\)                                    Bxh�]   
�          @��H����@�@l(�B�C�3����?n{@�\)B033C'��                                    Bxh�k�  "          @��R�\)@z�@mp�B\)C�=�\)?xQ�@���B8��C&T{                                    Bxh�zl  �          @�ff�h��@.�R@[�B
=C0��h��?���@�(�B:{CO\                                    Bxhŉ  
�          @����?�z�@o\)B!��C�=���>�{@��\B4
=C/33                                    Bxhŗ�  �          @�
=�l��?��@l(�B,{CY��l��=�G�@|(�B;=qC2G�                                    BxhŦ^  �          @�G��l(�>�
=@Y��B)�
C-�{�l(��8Q�@VffB&C>��                                    Bxhŵ  
�          @��R�w�?(�@=p�BQ�C+��w��Ǯ@?\)B�C9��                                    Bxh�ê  T          @�G��fff?�@Mp�B�C���fff?E�@j=qB3ffC'��                                    Bxh��P  .          @���8��@dz�@|(�B"Q�C ��8��@ff@�(�BW33C�                                    Bxh���  
�          @ȣ��Dz�@=p�@��RB9  C\�Dz�?�{@�Bd
=C&f                                    Bxh��  �          @�33�333@U�@��
B:��C#��333?�z�@�{BlffCn                                    Bxh��B  "          @\���@p  @�  B2(�B�ff���@�@�\)Bn�
Cs3                                    Bxh��  T          @�G���\)@C33@[�B7
=B�Q쿯\)?��
@�\)Bx{B���                                    Bxh��  �          @��\�O\)@QG�@E�B)ffB��
�O\)@
=@}p�Bq
=B�                                    Bxh�*4  �          @�{�\(�@!�@AG�BA��B�ff�\(�?�
=@j�HB�#�B��
                                    Bxh�8�  �          @����n{?8Q�@�HBz�C)�n{��Q�@!�B��C5h�                                    Bxh�G�  �          @�G��|(�?5@��A�G�C)���|(��#�
@��A�z�C4.                                    Bxh�V&  �          @�
=�h��?z�@33A�(�C*�R�h�ý���@�A��
C5��                                    Bxh�d�  T          @��H�`��?
=?�(�A�G�C*xR�`�׽�\)@33A�Q�C5�                                    Bxh�sr  
�          @�=q�o\)?�Q�?���A���C"c��o\)?��@ffA�ffC+��                                    BxhƂ  �          @�ff�tz�?��@G�A��HC$���tz�>�{@  A�33C.޸                                    BxhƐ�  
�          @���j�H?�ff@33A�(�C$��j�H>�{@�A��C.�f                                    BxhƟd  
�          @�  �i��?s33@�\A�  C%c��i��>8Q�@{BQ�C1&f                                    BxhƮ
  
Z          @���j�H?��@��A�p�C �f�j�H?�@\)Bp�C+��                                    BxhƼ�  T          @�33�o\)?�33@AظRC33�o\)?h��@   B�C&T{                                    Bxh��V  	�          @`  �(��?��?��HB
\)C��(��>���@��B�
C+u�                                    Bxh���  
�          @   ��?��?333A��C  ��?J=q?�  A��HC��                                    Bxh��  
�          @G��"�\?\?G�Alz�C  �"�\?���?�(�A�p�C��                                    Bxh��H  "          @)�����
?�Q�?8Q�A�\)C
���
?��?���A���CO\                                    Bxh��  
�          @Dz����?���?��A�33C8R���?�=q?�p�B
ffCT{                                    Bxh��  �          @R�\�{?�  ?��HA���C�3�{?��?�Q�A��RCz�                                    Bxh�#:  "          @�z��L(�?�z�@@  B"C��L(�>�
=@R�\B6�\C,�{                                    Bxh�1�  �          @����g�?�
=@r�\B3z�C"  �g��L��@~{B>p�C4�\                                    Bxh�@�  T          @��R��  ?�33@=qA��C\��  ?���@8Q�A�33C&�H                                    Bxh�O,  
�          @ҏ\��?���@���B��C$T{��=��
@�Q�B#G�C3�                                    Bxh�]�  �          @������H?���@|��B#�
C"�����H=��
@�p�B/ffC2�q                                    Bxh�lx  T          @���y��?z�H@n{B+33C%��y���W
=@uB2�C7
                                    Bxh�{  �          @�\)���
?n{@��HB0�\C'�q���
��G�@�p�B4�\C9�                                     Bxhǉ�  T          @�G��@��?�{��  ���HCW
�@��@33�������
C�
                                    Bxhǘj  �          @y���E�?�ff����ffC!)�E�?�G�>���@�
=C!�\                                    Bxhǧ  �          @Tz��7�?z�H?��A�\)C!
�7�?�R?��A���C'�=                                    Bxhǵ�  �          @Y���5?��\?���A�33C @ �5?(�?���A�Q�C'�f                                    Bxh��\  T          @�=q�hQ�<#�
@G�B G�C3�{�hQ�.{@
�HA�C>��                                    Bxh��  
�          @`  �4z��@ ��B  C6�=�4z�:�H?��Bz�CBu�                                    Bxh��  "          @~{�:�H>aG�@*=qB(�HC/�R�:�H�
=@&ffB$p�C?u�                                    Bxh��N  
�          @s33�N�R?xQ�>���@���C#E�N�R?W
=?z�Az�C%z�                                    Bxh���  T          @Dz��	��?�녿˅����C@ �	��?�ff������\)C#�                                    Bxh��  �          @X�ÿ��?�p�� ����B����@\)��=q����B�u�                                    Bxh�@  
�          @W
=�  ?@  ���1�C!}q�  ?�z��   �C�
                                    Bxh�*�  �          @Fff���?(���
�E�
C"
=���?��\�G��&��C�3                                    Bxh�9�  
(          @����p�=#�
�����*�RC2����p�?���G����C#Y�                                    Bxh�H2  
�          ?�׿����  ��p��(�RC>�����>���  �+��C.��                                    Bxh�V�  �          @����������.��C>xR��>#�
��33�1z�C-��                                    Bxh�e~  
4          @p�>�  ���@  ����C�N>�  ��\)���H�#�C�7
                                    Bxh�t$  
N          @�R�.{���H���033C����.{�z�H����t(�C��R                                    BxhȂ�  �          @#33?�p���>���AQ�C�ٚ?�p����;k�����C��)                                    Bxhȑp  "          @p�>����H?��RB6
=C��R>���=q?�  A��
C��                                    BxhȠ  T          @�p�@���hQ��������C��@���5�G
=���C�N                                    BxhȮ�  
�          @��@����r�\�?\)��ffC�J=@����0  �~{��C���                                    BxhȽb  �          @��
@�(��g���\)��33C���@�(��:�H�6ff��C��                                    Bxh��  �          @θR@��n�R��(��,��C�#�@��Mp��  ��=qC�q                                    Bxh�ڮ  
�          @��H@���X�ÿ�z��U�C���@���5�����(�C���                                    Bxh��T  �          @�G�@�z��o\)�\�V�HC�H@�z��H���"�\���RC�Q�                                    Bxh���  
�          @���@�z���33�����:�HC��q@�z��c33�������C�                                    Bxh��  
�          @�{@��R��zῪ=q�0z�C��\@��R�tz��"�\��C��                                     Bxh�F  T          @���@����zῥ��,z�C��@���u��   ��(�C��{                                    Bxh�#�  T          @ڏ\@���{�����]G�C�޸@���R�\�,����C�7
                                    Bxh�2�  T          @�p�@���s�
�����ffC��@���=p��X����ffC�u�                                    Bxh�A8  "          @�
=@�p���  �(���z�C�n@�p��Mp��P  ��C�s3                                    Bxh�O�  
Z          @��@������z���33C�R@���X���J�H�ݮC���                                    Bxh�^�  
�          @�{@�����\�����C���@���c33�N�R��ffC�Ff                                    Bxh�m*  T          @�Q�@�z���Q��G��r�\C�xR@�z��e��9����ffC��                                    Bxh�{�  �          @��
@�����(��Ǯ�L  C���@����`���*�H��p�C��=                                    BxhɊv  �          @�G�@����  �z�H����C��@���r�\�	����C�k�                                    Bxhə  
�          @�=q@�=q��=q�O\)�ҏ\C���@�=q����z���ffC���                                    Bxhɧ�  "          @�33@����=q����s33C�� @����Q�޸R�d��C���                                    Bxhɶh  
Z          @�@��H���H��=q�C�
C��3@��H�n{�/\)����C���                                    Bxh��  
�          @�  @�p��~�R�333��{C�xR@�p��A��s33���C��                                    Bxh�Ӵ  
�          @��@�  �Z=q�(���ffC��@�  �%�S33����C��                                    Bxh��Z  �          @�G�@ə��A녾����C�*=@ə��5������z�C���                                    Bxh��   �          @Ϯ@����A녿h����
C���@����)����(��{\)C�(�                                    Bxh���  
�          @Ǯ@�{�#33��z��v�\C�xR@�{��p��33��G�C���                                    Bxh�L  �          @��H@���6ff��  �S�C�Ф@����
�\)��
=C�f                                    Bxh��  �          @�\)@�33�(Q���R��(�C�� @�33��p��)�����C��=                                    Bxh�+�  
�          @ҏ\@�=q�z��G����C�8R@�=q��{�0����=qC�p�                                    Bxh�:>  |          AG�@��H����\)��C��)@��H����=q��C���                                    Bxh�H�  �          A�@�
=@33��ff�{AuG�@�
=@g
=��������A�z�                                    Bxh�W�  "          A\)@�33@   �\��Ak�
@�33@fff��G�� AʸR                                    Bxh�f0  T          A   @��H@������'�A���@��H@|�������G�A�R                                    Bxh�t�  �          A�@���@&ff�����3��A��@���@���G��B�\                                    Bxhʃ|  "          A\)@ҏ\>�33�����.p�@A�@ҏ\@�
��{�#  A��H                                    Bxhʒ"  �          A!p�@���@�
�أ��+
=A��R@���@u���ff�z�A���                                    Bxhʠ�  
�          A,��@��@����\)�7{BG�@��@������Q�B/z�                                    Bxhʯn  "          A)�@Ǯ@�G����4�
B�R@Ǯ@��Å�\)B.(�                                    Bxhʾ  
�          A'\)@�\)@w���33�B��B33@�\)@����  �B6��                                    Bxh�̺  
�          A(��@��@����{�B�HB�@��@����G���B;�H                                    Bxh��`  	�          A'33@��
@�  ���
�C�
Bp�@��
@�
=��p��  BD�
                                    Bxh��            A#�
@�Q�@�Q����W(�B3�H@�Q�@�=q�أ��&\)B`{                                    Bxh���  �          A(  @�{@�\)����J�BA(�@�{@�\)�θR��HBf=q                                    Bxh�R  
�          A%��@���@l(����\�E  B��@���@���љ��33B4Q�                                    Bxh��  T          A!��@�=q@8����  �8�HA�Q�@�=q@�\)�ƸR��RB=q                                    Bxh�$�  T          A (�@��
@�H�����3��A��H@��
@�
=���
��B{                                    Bxh�3D  "          A��@�Q�?�(�����7��A�
=@�Q�@q���\)��A�\)                                    Bxh�A�  
�          Ap�@�=q?�G��ə��$G�A@��@�=q@H����p���HA�p�                                    Bxh�P�  
�          A"=qA=q�����\)���C�{A=q?��H��33�
=A33                                    Bxh�_6  
�          A(��Az�ٙ�����33C�˅Az������
=���C��f                                    Bxh�m�  �          A)G�A�R������=q����C��A�R��Q���Q���ffC��)                                    Bxh�|�  T          A(��A��?G�������@�{A��@	����(���AVff                                    Bxhˋ(  
�          A*�\A�R?+�������@��A�R@
�H�����=qAZ�\                                    Bxh˙�  �          A+
=Aff�+������RC���Aff?G������  @�z�                                    Bxh˨t  
Z          A�\A����Q�������C��qA��?s33���\��33@�ff                                    Bxh˷  �          A	G�@ָR?��������G�A"=q@ָR@\)����{A��R                                    Bxh���  r          Az�@�p�?�\)��33�-{A�{@�p�@U������\A�G�                                    Bxh��f  "          A��@�(�@Q���ff�N�A�(�@�(�@p  ����.(�B�                                    Bxh��  
Z          A��@��@;����R�5G�A�
=@��@�z���\)��B"                                    Bxh��  
�          @Dz�@#�
�z�\(���G�C��)@#�
��{�z�H���RC�E                                    Bxh� X  �          @�{@^�R��;L���/\)C�ٚ@^�R�z�B�\�*�HC��q                                    Bxh��  "          @��@>{�A�?��RA��HC�:�@>{�O\)>��@�(�C�9�                                    Bxh��  T          @�33@C33�:�H?�  AU��C�  @C33�E�>W
=@0��C�\)                                    Bxh�,J  
�          @��
@`  � ��?�RA�C�8R@`  �%��L�Ϳ#�
C�Ф                                    Bxh�:�  
�          @���@{��
=q�Ǯ����C��q@{���p��p���@��C��q                                    Bxh�I�  
�          @�p�@�\)��\�Y���(��C�� @�\)���R���
�\)C�G�                                    Bxh�X<  "          @��\@�
=�Ǯ��  ��33C�� @�
=��ff���33C��
                                    Bxh�f�  �          @�\)@���G��K���
=C�+�@������Y�����C��\                                    Bxh�u�  
�          A
ff@�
=���������C�ff@�
=�(����H�=qC���                                    Bxh̄.  
�          @�{@ƸR��
=��{�p�C��=@ƸR�.{�����
=C�1�                                    Bxh̒�  �          @���@�����\)��p��{C�T{@��������z��ffC��
                                    Bxh̡z  
�          Aff@�녿�=q��  ��ffC�|)@�녾�G���Q���C�(�                                    Bxh̰   �          @߮@�G��޸R�a���ffC�� @�G��G��w
=���C�%                                    Bxh̾�  
�          @�
=@Ǯ���^{�݅C�Z�@Ǯ��p��z=q���RC�n                                    Bxh��l  T          A ��@���7��@  ���HC��
@��� ���hQ���Q�C��                                     Bxh��  �          @�Q�@��R�e���
�l(�C���@��R�AG��(Q���33C��                                    Bxh��  
�          @�Q�@���ff�����R{C�c�@����Q����  C�S3                                    Bxh��^  "          @أ�@����,�Ϳ\�O\)C��@�����R�	����Q�C���                                    Bxh�  T          @׮@�ff�:=q�����$��C��
@�ff� �׿�33��33C��                                    Bxh��  
�          @�z�@�33�X�ÿ�\)��C�ff@�33�@  ��Q����
C���                                    Bxh�%P  
�          @�Q�@���P�׿u���C�@���:�H��  ��C�t{                                    Bxh�3�  �          @�{@��\�u<�>���C�@��\�n�R�n{�
=C�.                                    Bxh�B�  
�          @���@z=q�;�>\@��RC��\@z=q�;���p�����C���                                    Bxh�QB  
�          @Ǯ@�ff�p��>��@��C�aH@�ff�mp��0���˅C���                                    Bxh�_�  "          @��
@�=q�e?
=@��C��=@�=q�g����
�E�C���                                    Bxh�n�  T          @�p�@��
�i��>�G�@�Q�C�޸@��
�h�þ���ffC���                                    Bxh�}4  �          @��\@Tz��\��?�G�A�(�C��=@Tz��n{?��@��C�ٚ                                    Bxh͋�  �          @R�\@�H���R?aG�Az{C�L�@�H���>�33@ȣ�C�<)                                    Bxh͚�  �          @�\)@Dz��mp�?�p�Aep�C��\@Dz��y��>�=q@H��C�)                                    Bxhͩ&  �          @�(�@8Q����?˅A���C�{@8Q���  >��H@��RC�K�                                    Bxhͷ�  "          @���@)������?�A��HC�1�@)�����\?#�
@���C�b�                                    Bxh��r  
�          @�\)@ff����@�
A��C��@ff��z�?Q�A�C�J=                                    Bxh��  "          @��R@{���
@�RA�=qC�ٚ@{���H?�G�AL��C��                                    Bxh��  
�          @�G�@   ���@4z�A��C��@   ��p�?�z�A�=qC�p�                                    Bxh��d  
�          @��\?�z��8��@%�BQ�C���?�z��\(�?��A��
C�K�                                    Bxh�
  "          @�(�?��
���@n{B�C���?��
����@$z�A�z�C�y�                                    Bxh��  �          @\?5����@�G�B@(�C�H?5��Q�@Z�HB33C��                                    Bxh�V  	�          @��׽��
�QG�@�  Bc\)C�E���
��z�@���B*�
C�t{                                    Bxh�,�  �          @����G��&ff@��Bfz�Co�f��G��k�@}p�B3�Cw�R                                    Bxh�;�  �          @�33��  ��@q�B(Q�C�K���  ��\)@)��A�{C���                                    Bxh�JH  
�          @�=q?@  ���@]p�BQ�C�=q?@  ��  @Q�A�33C��)                                    Bxh�X�  
�          @�G�@����  ?��A!�C�H@����{>aG�?�p�C�j=                                    Bxh�g�  T          A\)@4z���p�@+�A�\)C�h�@4z���33?L��@�Q�C��                                    Bxh�v:  T          Az�@�=q���?�(�A;�C���@�=q���
    ���
C�Z�                                    Bxh΄�  
�          A�H@�Q����Ϳ����C�&f@�Q����R�{��ffC�P�                                    BxhΓ�  �          A�?������@�  B+C�y�?���(�@�z�A�
=C���                                    Bxh΢,  
�          A>�\)�θR@�\)B8�C�:�>�\)��z�@�\)B �
C�H                                    Bxhΰ�  T          A=q�
=��  @��BDC�H��
=��=q@��BQ�C��3                                    Bxhοx  T          A�
�\(���@�33BK33C�ٚ�\(����@��RB�C��R                                    Bxh��  �          A녿��
���@��
BXp�C}^����
��\)@�B"C��)                                    Bxh���  
�          A�R��(���
=@�B[�\Cz녿�(����
@\B&z�Cn                                    Bxh��j  
�          A��
=q��
=@��BZ�Cun�
=q�ʏ\@�
=B'{C{.                                    Bxh��  
�          A���\��  @�{BH�
C|&f��\��  @�BC��                                    Bxh��  
�          Aff�s33��(�@�=qB#��C����s33���\@�=qA�Q�C��f                                    Bxh�\  
�          A{�G���\@���B
=C��R�G��{@\(�A��\C�@                                     Bxh�&  "          A��<#�
��\@��B\)C��<#�
���@U�A���C�
=                                    Bxh�4�  
�          A	�?8Q���ff@�p�A���C��?8Q���@�Aw\)C���                                    Bxh�CN  
Z          A�?G����@��A�ffC���?G��ff@Q�Ab�RC��=                                    Bxh�Q�  
�          A��?�G��Q�@g
=A�G�C�y�?�G��=q?�ffAG�C�<)                                    Bxh�`�  T          A�>���\)@g
=A��C�n>���G�?���A ��C�T{                                    Bxh�o@  �          A(�>L���@Tz�A���C���>L����\?�  @��C��H                                    Bxh�}�  T          A�?   �z�@W�A��C��
?   �p�?�=qA��C��)                                    Bxhό�  	�          A=q?333�{@aG�A��\C�xR?333��?�G�AQ�C�L�                                    Bxhϛ2  	�          A\)?B�\�   @UA��\C���?B�\���?�\)A��C���                                    Bxhϩ�  T          Ap�?�{�G�@S�
A�  C�Ǯ?�{�
{?���A	C�z�                                    Bxhϸ~  �          A  ?У���33@`��A��HC�� ?У��33?���A%p�C�w
                                    Bxh��$  T          A��>�����@���A�p�C��R>���  @�RAj�RC��                                    Bxh���  �          A�>�  ��33@�  B 
=C��>�  �  @6ffA�(�C���                                    Bxh��p  T          Aff?(�����R@��
B C�o\?(���
=q@<(�A�  C�,�                                    Bxh��  �          A
=?.{� ��@��
A�G�C�g�?.{���@Q�Ao�
C�0�                                    Bxh��  
�          Aff?h���33@QG�A���C�,�?h����
?��\A�C���                                    Bxh�b  
�          A�?��R��@VffA�G�C�Y�?��R�
�H?�\)A{C�3                                    Bxh�  
�          A��?�\)� ��@QG�A���C��3?�\)�	p�?���A
�\C��{                                    Bxh�-�  T          A	G�?h��� Q�@5�A��\C�=q?h���33?fff@�Q�C�3                                    Bxh�<T  
�          Az�?z�H��
=@3�
A�Q�C���?z�H��\?c�
@���C�U�                                    Bxh�J�  
�          A	G�?������
@l��A�(�C��?����Q�?�=qAE��C��                                     Bxh�Y�  �          A��?��R��@^{A�
=C�|)?��R�z�?���A,Q�C��                                    Bxh�hF  
�          A��?��R��{@Z�HA��HC�� ?��R�Q�?ǮA(Q�C��                                    Bxh�v�  
�          AQ�?��\��
=@VffA�p�C��=?��\�z�?�p�A ��C�Z�                                    BxhЅ�  �          A�?�G���R@��A��C���?�G���ff@33A}C�`                                     BxhД8  
�          A
=?��
���
@�33A�C��?��
���
@
=A�Q�C�b�                                    BxhТ�  
�          A��?�����@��A��C��f?����@+�A��HC��H                                    Bxhб�  
Z          A	�@���(�@�33A��C�,�@���ff@*=qA��\C�B�                                    Bxh��*  
�          A�?��R��
=@�A�
=C�
=?��R����@.{A�\)C�j=                                    Bxh���  T          A�H?�(���G�@��HA���C��?�(���33@'�A�33C�N                                    Bxh��v  T          A�?�����
@^{A�=qC�%?����
=?ٙ�A;�C��                                     Bxh��  
Z          A��@Q�����@Z�HA���C��@Q��?�{A.{C�]q                                    Bxh���  
�          A	�@��\)@N{A���C��3@�(�?�\)A=qC�)                                    Bxh�	h  �          A@ �����@p  A�p�C��R@ ����{@33Ac�
C��                                    Bxh�  �          A��@?\)��{@�  A�
=C��@?\)��@Q�A�Q�C��
                                    Bxh�&�  �          Ap�@����(�@��A�ffC��@�����
@��A�Q�C�B�                                    Bxh�5Z  �          A��@z���  @�Q�B�C��R@z����@J=qA���C���                                    Bxh�D   
�          Ap�?����\)@��\A�
=C�<)?������@(��A�  C��{                                    Bxh�R�  T          A��?\��z�@S�
A��C��3?\��{?ǮA.ffC�l�                                    Bxh�aL  
�          A��@�  ���\��p��Q�C���@�  ����p����C���                                    Bxh�o�  
�          A33@�
=�'
=��\)��G�C�^�@�
=�˅���R�	=qC�T{                                    Bxh�~�  
Z          A�@�G��j�H�Vff���HC�=q@�G��4z����H��\C��                                    Bxhэ>  T          A��@�
=�w
=�P  ��
=C���@�
=�A�������  C���                                    Bxhћ�  T          A@�Q��|���J�H���C�� @�Q��HQ��~�R�ՅC�W
                                    BxhѪ�  
g          A�@�\�tz��0  ��\)C�<)@�\�Fff�b�\���
C��f                                    Bxhѹ0            A  @��
�q��4z���ffC�k�@��
�C33�fff��(�C��                                     Bxh���  �          A�@��
�~{�����C��R@��
�Tz��P����C��q                                    Bxh��|  T          A33@��
��p����H�P��C�=q@��
�hQ��6ff����C��                                    Bxh��"  
�          A
ff@�G���z����Yp�C�1�@�G��e��9����G�C���                                    Bxh���  �          A
�\@�{�fff�Mp���Q�C��
@�{�2�\�|(���Q�C�h�                                    Bxh�n  
�          A�R@�ff�k��;����C��@�ff�<(��k���  C���                                    Bxh�  
�          A
=@�  �s�
�.{���HC��
@�  �G
=�`  �ģ�C��                                    Bxh��  
�          A��@ڏ\��p���  �'\)C���@ڏ\�~{�����C��f                                    Bxh�.`  
�          A��@ۅ���
������C���@ۅ��Q���\�f�\C��
                                    Bxh�=  "          A(�@�z��|(��G��
=C�  @�z��U�E���C��                                    Bxh�K�  �          A{@����\�k��ə�C�'�@���  ���eC�3                                    Bxh�ZR  �          A�H@�G�����8Q쿜(�C�#�@�G������33C���                                    Bxh�h�  
�          A\)@�Q���z�����,(�C�,�@�Q���p���=q�,��C���                                    Bxh�w�  
�          A�@�����zᾙ����\C�:�@�����{���R�"{C�                                    Bxh҆D  
          A(�@�ff��G���=q����C���@�ff��33��  �!�C�,�                                    BxhҔ�  �          A{@�\)���H>�{@C��)@�\)���׿u�ҏ\C��                                    Bxhң�  
�          A33@����?���@�  C�U�@����Q쾣�
�\)C�%                                    BxhҲ6  
�          @�@������?��@��HC��R@����33���R��C���                                    Bxh���  �          A ��@\)��(�@�At��C���@\)��{?��@���C��3                                    Bxh�ς  T          A (�@|(����H@z�Aq�C�o\@|(���(�?\)@���C���                                    Bxh��(  
Z          A ��@{���{@0  A�Q�C��3@{���z�?��\A�C���                                    Bxh���  
Z          A ��@G���\)@x��A�C�J=@G���{@�A�p�C��                                    Bxh��t  "          A   ?�����H@��HBQ�C�,�?����  @XQ�A�z�C�&f                                    Bxh�
  �          @�{?�33���H@�B{C��?�33����@a�A���C��{                                    Bxh��  �          @�ff@   ��{@�G�B�
C�  @   ��\)@4z�A��RC���                                    Bxh�'f            @�ff@5�����@|(�A��C�#�@5��߮@p�A��\C��                                    Bxh�6  T          @�@E��
=@W�A��C�@E�ᙚ?�{AY��C��{                                    Bxh�D�  "          @��@�G����H@i��Aٙ�C�W
@�G��Ϯ@�A���C��\                                    Bxh�SX  �          @�\)@�������@W
=A�C��@����Ϯ?�p�Ag
=C�H�                                    Bxh�a�  "          @�{@~�R���@_\)A��HC���@~�R����@
=Aw
=C���                                    Bxh�p�  �          @�p�@�����@.�RA��
C���@���Ӆ?��Ap�C��                                    Bxh�J  
�          @�ff@�=q�ƸR@=qA�33C�k�@�=q���H?}p�@�p�C���                                    BxhӍ�  �          @��@�z����H@  A��\C���@�z���?O\)@���C���                                    BxhӜ�  �          @�z�@��
����?��
A333C�t{@��
��
=>8Q�?�ffC�                                    Bxhӫ<  
�          @�33@�\)���?��A z�C��\@�\)���;��R��\C��H                                    Bxhӹ�  �          @��\@g�����@(��A�ffC�=q@g��ҏ\?��RAp�C�j=                                    Bxh�Ȉ  T          @��@7���(�@���B��C�f@7���p�@9��A��C���                                    Bxh��.  �          @��@e��33@��A��C�\)@e�˅@333A���C��q                                    Bxh���  �          @�p�@7
=����@�(�BffC��\@7
=�θR@c�
A��C��3                                    Bxh��z            @�p�@�\��{@�{B�C���@�\���@Q�A�ffC�L�                                    Bxh�   T          @�ff@(Q���
=@�33Bp�C�XR@(Q���(�@_\)A�Q�C��                                    Bxh��  
�          @�z�@x����@fffA�\C�s3@x�����H@��A�ffC���                                    Bxh� l  �          @��@�p���ff@�
Av�RC�޸@�p�����?���A=qC���                                    Bxh�/  x          @��@�z���G�?��RAk�C�R@�z��\?.{@�Q�C�ff                                    Bxh�=�  T          @��
@�33��  ?��HAJ�\C�j=@�33��\)>��R@�\C���                                    Bxh�L^  �          @��\@�=q���H?��A`Q�C��=@�=q�Å?z�@�  C�'�                                    Bxh�[  "          @��H@�  ��G�@��A��RC�b�@�  ��{?�Q�AQ�C�Z�                                    Bxh�i�  T          @��\@������@z�Au�C��@�����?n{@ڏ\C�#�                                    Bxh�xP  "          @�G�@�G����?�G�A2�HC�(�@�G���(�>�@X��C��f                                    BxhԆ�  �          @��
@ə����\?aG�@�{C��@ə�������xQ�C��                                    Bxhԕ�  "          @���@��H��33?�A��C���@��H���=�Q�?(��C�T{                                    BxhԤB  T          @�@����=q@J�HA�{C�xR@����(�?���Aip�C�\                                    BxhԲ�  
�          @��R@������@���B�\C�9�@�����
=@U�A�Q�C�
                                    Bxh���  
�          A z�@l����ff@��\B=qC�
@l�����@R�\A�C�0�                                    Bxh��4  
�          A ��@Vff��=q@���A�z�C�t{@Vff�ٙ�@(��A�C�q                                    Bxh���  T          @�\)@~�R��z�@�=qB=qC�7
@~�R��ff@C33A�  C�Y�                                    Bxh��  �          @�
=@�
=����@w�A�\C��=@�
=��\)@%�A�G�C��                                    Bxh��&  "          @��@�
=��=q@|��A���C�f@�
=����@,��A�=qC�=q                                    Bxh�
�  "          Ap�@�(����@�z�A��C�H�@�(���=q@5A�
=C��\                                    Bxh�r  
�          A@��
���
@g
=A�p�C��@��
����@
=A��C�|)                                    Bxh�(  �          A�@����33@w�A�RC���@������@'�A�p�C���                                    Bxh�6�  �          Aff@�\)��G�@w�A���C��@�\)�Ϯ@!�A��\C��\                                    Bxh�Ed  T          A�@x������@o\)A�  C�b�@x���ָR@ffA���C�                                    Bxh�T
  
�          A�@l�����H@^�RA��HC�!H@l����{@�Ag\)C�f                                    Bxh�b�  
(          A=q@�����  @P��A�33C�}q@������?�AP��C�e                                    Bxh�qV  
�          A�@y���ȣ�@QG�A�\)C���@y���ڏ\?�AS
=C�ٚ                                    Bxh��  T          AG�@n�R�ȣ�@\(�Aə�C�^�@n�R���
@   AeC�AH                                    BxhՎ�  �          A�@���Ǯ@?\)A�=qC���@���׮?���A3�
C���                                    Bxh՝H  
�          A��@tz���@��A�33C�� @tz��ᙚ?s33@�Q�C�7
                                    Bxhի�  
�          AG�@�����
=@<(�A�p�C��)@����θR?˅A5C��f                                    Bxhպ�  T          A ��@�����@AG�A��C���@���G�?�G�AK33C��q                                    Bxh��:  T          A ��@����G�@�z�B{C�ٚ@�����
@L��A��C���                                    Bxh���  �          A ��@�{���@�G�A�
=C�
=@�{��Q�@A�A�ffC���                                    Bxh��  �          A��@�  �p��@�G�A��C�c�@�  ��33@XQ�A�33C���                                    Bxh��,  �          A��@�G�����@���A�p�C��@�G����@@��A�
=C�Ǯ                                    Bxh��  
�          Ap�@�����@c33AϮC��
@�����R@z�A���C���                                    Bxh�x  �          A=q@�  ��{@Q�A��RC�e@�  ��G�@(�Az�RC���                                    Bxh�!  �          A�@�����\)@5A��C��)@�����\)?�AN{C�~�                                    Bxh�/�  "          A{@�33���@�RA�{C���@�33��\)?�=qAffC�l�                                    Bxh�>j  
�          A ��@��H����@7
=A���C��\@��H��Q�?�z�A>�\C�g�                                    Bxh�M  
�          Ap�@�����{@'�A�\)C��@�������?��A0Q�C��q                                    Bxh�[�  �          A��@�
=���
@0��A��C��f@�
=��33?��A=�C�p�                                    Bxh�j\  
�          A@������@e�A�p�C�}q@������@��A��RC��=                                    Bxh�y  
�          A Q�@�p����@9��A�p�C�ٚ@�p�����?�z�A?�C��
                                    Bxhև�  �          @��@��
��G�@(Q�A�p�C�\)@��
��\)?�\)A��C�H�                                    Bxh֖N  �          @�
=@��H���R@z�Ap��C���@��H�ȣ�?:�H@���C���                                    Bxh֤�  "          @���@�33��\)?��
AP��C��
@�33��\)?   @hQ�C�<)                                    Bxhֳ�  
�          @�z�@�=q��Q�?�G�AN�RC��=@�=q��Q�>��@]p�C�3                                    Bxh��@  
�          @���@����(�?���A:�RC��@����33>�33@   C��                                    Bxh���  �          @�(�@��H���\?��RA.{C�޸@��H����>�  ?��C�e                                    Bxh�ߌ  T          @�z�@�
=��z�?�AD(�C�
@�
=�Å>�Q�@(��C���                                    Bxh��2  
�          @�33@�Q����?��HAH��C�<)@�Q���
=>�p�@-p�C��R                                    Bxh���  "          @��\@�Q����
?ǮA8(�C���@�Q����>��?�=qC���                                    Bxh�~  
Z          @���@�p��ə�?�
=AEp�C��@�p��У�>�\)@33C��                                    Bxh�$  �          @�=q@��
��33?�
=A�C�C�@��
��ff�aG��˅C��                                    Bxh�(�  "          @�@�=q����?�Q�AEC�S3@�=q��  >�33@!G�C���                                    Bxh�7p  
�          @��@����Ϯ?�G�@��
C�\@�����녾���@  C��                                    Bxh�F  F          @�z�@~�R�ָR?}p�@�RC�XR@~�R��Q���H�c33C�>�                                    Bxh�T�  �          @�{@�33����?�33A`  C���@�33��p�?z�@�  C�R                                    Bxh�cb  
Z          A (�@�Q��ƸR?ٙ�AD��C��)@�Q���>��
@33C�@                                     Bxh�r  "          @�
=@����
=?˅A8��C�ٚ@�����>��?���C�w
                                    Bxh׀�  b          A Q�@�Q���(�?�(�A,(�C��\@�Q���G�        C�B�                                    Bxh׏T  
�          A�R@�ff��\)@>�RA�G�C�3@�ff�Ϯ?�\)A6ffC���                                    Bxhם�  "          A@�����@K�A�
=C�u�@���33?���AS33C�33                                    Bxh׬�  �          AG�@�����33@0  A�G�C�O\@����љ�?�{A�C�W
                                    Bxh׻F  "          A\)@�33�ȣ�@)��A�ffC�q@�33��ff?�(�A	�C�:�                                    Bxh���  �          A�R@�=q���@���A�C�
=@�=q��G�@C33A���C�H                                    Bxh�ؒ  �          A(�@��
��33@��BG�C���@��
��G�@n�RA�C��q                                    Bxh��8  T          A�@�ff��@���B.p�C�/\@�ff�O\)@�{BffC�4{                                    Bxh���  "          A�R@�=q��z�@���B-�C��@�=q�G�@�
=B33C��                                    Bxh��  �          AG�@�\)��ff@��\B1�C�t{@�\)�1�@�33B 33C��                                    Bxh�*  "          A=q@����ff@���B7�C���@����@���B)�C���                                    Bxh�!�  
�          Ap�@�(��W
=@��BD�
C���@�(��(�@�\)B7G�C��\                                    Bxh�0v  �          A
ff@�p��0��@�Q�B9�C���@�p���\@��RB.ffC��                                     Bxh�?  "          A	@��ÿaG�@�=qB3��C��q@�����@�\)B'G�C�3                                    Bxh�M�  �          A��@�33��{@���B)��C���@�33�5�@���B��C���                                    Bxh�\h  
�          A	G�@�G���{@�G�B)�
C�s3@�G��E�@��B��C���                                    Bxh�k  �          AQ�@�p���
=@��
B.��C�3@�p��:�H@��
B\)C�޸                                    Bxh�y�  �          A  @����ff@��\B-33C��f@���A�@�G�B�C�n                                    Bxh؈Z  T          A�R@��
��R@�Q�B,�C��@��
�k�@��\Bz�C�k�                                    Bxhؗ   
�          A��@�33��z�@��\B=qC��)@�33����@\��A��C�Z�                                    Bxhإ�  
�          A�R@�����R@�B�HC�{@����(�@eAиRC�h�                                    BxhشL  T          A��@�33���
@���B�RC���@�33���@l��A��C�33                                    Bxh���  T          A ��@������@�(�B��C��
@����@e�AӅC��
                                    Bxh�ј  �          A ��@�Q���G�@�(�B33C���@�Q���ff@dz�A���C���                                    Bxh��>  �          AG�@��
��
=@��B  C�E@��
��(�@dz�A�C��                                     Bxh���  �          AG�@������@�(�B!\)C�k�@�����R@�\)A���C���                                    Bxh���  T          A
=@�z���{@�\)B"  C���@�z�����@�=qA�(�C��
                                    Bxh�0  �          AQ�@������\@��RB��C�#�@�����p�@���A��C�N                                    Bxh��  �          A�@�������@�p�B��C��@�����
=@��RA�=qC���                                    Bxh�)|  "          A�H@�\)��z�@��RB��C�@ @�\)���@�  A���C���                                    Bxh�8"  
(          AG�@������@�G�B�C�@����Q�@w�A�G�C�W
                                    Bxh�F�  
�          A@�
=��\)@�Q�B
=C�� @�
=���@c33A��C�T{                                    Bxh�Un  
�          A{@����{@��
B
=C���@���\@W
=A��C��)                                    Bxh�d  
�          A��@�
=��(�@�ffB�C��q@�
=��\)@I��A�z�C��
                                    Bxh�r�  
�          @���@G����@7�A�{C�AH@G����?�ffA�
C��H                                    Bxhف`  �          @�=q?(����  @'�A���C��\?(�����?�G�@��C���                                    Bxhِ  �          @�ff@L����  @p�A��C��@L����=q?8Q�@�C���                                    Bxhٞ�  "          A z�@z���G�@*�HA�  C�q@z���
=?�(�A33C���                                    Bxh٭R  
Z          A33@����=q@���A�33C��{@����z�@C33A�
=C��\                                    Bxhٻ�  T          AQ�@�{��Q�@�G�A��C�8R@�{��Q�@'
=A��HC��f                                    Bxh�ʞ  �          A33@�Q����H@���A��HC��R@�Q����H@(Q�A��
C�4{                                    Bxh��D  �          A{@�����
=@��HA�C���@������@Dz�A�(�C�xR                                    Bxh���  �          AG�@�{��\)@�A�p�C���@�{����@:�HA�(�C��                                    Bxh���  �          A�H@=p����@A�A���C���@=p����?�A!�C��                                    Bxh�6  T          A
=@G
=��\)@{A{�C��f@G
=��G�?��@w
=C�7
                                    Bxh��  �          Aff@^�R�׮@9��A�G�C��=@^�R��R?���A{C��                                    Bxh�"�  
(          A Q�@E���ff@C�
A�ffC�U�@E���R?��RA,(�C���                                    Bxh�1(  
�          A (�@5�߮@�A�33C�
=@5��33?=p�@���C��\                                    Bxh�?�  "          Az�@H����{@\)A���C�˅@H�����?Q�@�{C�Ff                                    Bxh�Nt  
�          A�@w
=��Q�@)��A��RC�j=@w
=��p�?��\@�C���                                    Bxh�]  "          A�H@�ff��
=@p��A���C�f@�ff���@G�A{
=C��=                                    Bxh�k�  �          A�@�p���  @j=qA�G�C�/\@�p���p�@{Axz�C��                                    Bxh�zf  T          A�H@�Q���
=@�
=B\)C�(�@�Q���33@L(�A�G�C��)                                    Bxhډ  
�          A\)@����z�@�z�B��C���@������@W
=A��C�b�                                    Bxhڗ�  �          A{@��R��\)@�\)B
�\C�4{@��R��p�@c33A��
C���                                    BxhڦX  T          A�@�����{@���A��C�Ff@�����Q�@8��A�Q�C�n                                    Bxhڴ�  "          A��@����
=@��HB	{C��@����
=@j�HA�C�Ф                                    Bxh�ä  
�          A
=@�����@��\A��C�J=@������@Dz�A�p�C�xR                                    Bxh��J  "          AG�@�
=��G�@��HB.�
C�K�@�
=����@��BffC��{                                    Bxh���  �          AG�@����33?��A:ffC�Y�@����=u>�p�C�                                      Bxh��  �          AG�@�(�����@?\)A��C��@�(����?���A�
C�"�                                    Bxh��<  �          A�\@����p�@mp�A�\)C��\@�����H@
=qAZffC���                                    Bxh��  
�          A��@�  �ə�@dz�A���C�0�@�  ��ff@33AL��C��                                    Bxh��  �          A�\@��H��p�@���A��
C��
@��H��@"�\Az�HC��                                    Bxh�*.  �          Aff@��
��Q�@�{A�  C���@��
�ᙚ@*=qA��C�K�                                    Bxh�8�  
�          A��@�=q��
=@�=qA�G�C���@�=q��\)@\)Ax(�C�.                                    Bxh�Gz  "          A��@������
@�ffA�C�aH@�������@K�A�G�C�q�                                    Bxh�V   F          A@�
=���H@�  B	�C�y�@�
=���H@i��A�
=C��                                    Bxh�d�  �          A\)@�����\@��HB(�C��)@����p�@��HA�Q�C��{                                    Bxh�sl  �          A�@�33����@�
=B��C��)@�33����@p  A�p�C���                                    Bxhۂ  �          A=q@�\)����@�A�G�C�K�@�\)���H@3�
A�z�C�b�                                    Bxhې�  
�          @�Q�@\)����@g
=A��C��@\)��  @�HA��HC�                                      Bxh۟^  T          @�\)@qG��z�@7
=BQ�C��f@qG��*�H@�
A�=qC�aH                                    Bxhۮ  
�          @�=q@�Q���@'�A��HC�(�@�Q��>{?��RA��C���                                    Bxhۼ�  T          @�33@��
��@'
=A�z�C�)@��
�>{?�p�A��RC���                                    Bxh��P  �          @���@��
�9��@#�
A�G�C���@��
�Z�H?�=qA��C�                                    Bxh���  
�          @�G�@�=q�G
=@%A\C��
@�=q�hQ�?�A�p�C���                                    Bxh��  �          @���@�녿�p�@#33A��HC��)@���G�@�A��
C��H                                    Bxh��B  �          @Ǯ@������@1G�A���C�P�@����>{@��A��RC��                                    Bxh��  �          @�Q�@��H�J�H@2�\A�(�C�e@��H�n�R?�p�A��C�1�                                    Bxh��  �          @��
@$z�����@'�A�ffC��3@$z���  ?�=qA>ffC�f                                    Bxh�#4  �          @陚?��
��  @:=qA�z�C���?��
�߮?���A'�C�.                                    Bxh�1�  �          @��
?��
��p�@.{A�G�C�w
?��
���
?��A��C��                                    Bxh�@�  �          @��
@\)��  @*=qA�\)C��H@\)��{?�=qA�\C��                                    Bxh�O&  �          @�@\)��(�@ ��A��C�T{@\)����?fff@�{C��{                                    Bxh�]�  T          @�@ff��\)@ffA�\)C��f@ff���H?5@�Q�C�<)                                    Bxh�lr  �          @�\?�33��
=@{A�
=C��?�33�ᙚ?
=@��C���                                    Bxh�{  �          @�p�?���p�?��HAup�C�S3?���{>�z�@  C��                                    Bxh܉�  �          @�(�?�����?ǮAK
=C��)?����ۅ�L�;�p�C��=                                    Bxhܘd  �          @�\)>�33��
=@!�A��RC�� >�33���
?aG�@ᙚC�j=                                    Bxhܧ
  �          @�\)@=p���Q�?@  @ڏ\C�q@=p����ÿ�����C��                                    Bxhܵ�  �          @�z�@Tz���\)�L�;�G�C�@Tz���녿�33�B�RC�o\                                    Bxh��V  �          @ָR@S�
��녾�=q��C��{@S�
���H�У��a�C�T{                                    Bxh���  �          @��@L�����<��
>W
=C�+�@L����p����4z�C��f                                    Bxh��  �          @��\@ �����?
=q@���C�!H@ ������z���Q�C�#�                                    Bxh��H  �          @�z�?�{��G�?���Aep�C��R?�{���R>��?�  C�W
                                    Bxh���  �          @��\?.{���
?�33A]G�C�5�?.{��Q�<��
>�\)C��                                    Bxh��  
�          @���>����(�?��\A<Q�C���>����\)�#�
���C���                                    Bxh�:  T          @��
��������?��
A��C��)������G�?
=q@�Q�C���                                    Bxh�*�  �          @����(����  @�\A��C�B��(������?��
A+�
C���                                    Bxh�9�  �          @�  �k�����?�(�A���C��R�k����?8Q�@�G�C�&f                                    Bxh�H,  �          @�Q�>aG���  @
=qA�\)C�Y�>aG���(�?xQ�A*=qC�@                                     Bxh�V�  �          @���>�����?�=qA�33C��H>���Q�>���@e�C���                                    Bxh�ex  �          @�\)�����  @Q�A���C|�������
?n{A{C}B�                                    Bxh�t  T          @�(������
?�(�A��HCq�\����
=?\(�A��CsW
                                    Bxh݂�  �          @�z�����Q�?�=qA���Cp��������?�\@�
=CrG�                                    Bxhݑj  �          @���+���\)?��HAY��Cm���+�����>#�
?�\Cn��                                    Bxhݠ  �          @�\)�\)��=q?\A��
Cp
�\)���>�G�@�=qCqh�                                    Bxhݮ�  �          @�z�����  ?�p�A��Cr@ ������?
=@��Cs��                                    Bxhݽ\  T          @��
=q��ff?�p�A���C�� �
=q��G�?J=qA�C��
                                    Bxh��  T          @�p��
=q��{?�  Ak�
C�E�
=q��(�=��
?O\)C�\)                                    Bxh�ڨ  �          @��H�\)��{?��A/�C�\�\)���þ�z��=p�C�)                                    Bxh��N  �          @���0�����H>�@�{C��f�0�����׿��\�(�C��q                                    Bxh���  �          @�z���ə�?p��A33C�Ф�����H��R���C��{                                    Bxh��  �          @��ÿ������=L��>�C�~�������ff�����S�C�P�                                    Bxh�@  �          @�p���Q����W
=��CvQ��Q����H��33���Ct��                                    Bxh�#�  �          @�Q쿢�\��ff?W
=A,��C}!H���\���þW
=�.�RC}k�                                    Bxh�2�  �          @�Q�\)���H?=p�A"ffC�3�\)���;�\)�w
=C�                                      Bxh�A2  �          @�
=��  ���H�0�����Cy.��  ���ÿ�����Cw��                                    Bxh�O�  �          @�z�G�����>\)?��C�\)�G���{����EC�AH                                    Bxh�^~  �          @�����H���O\)�#�Cw�=���H�w
=������z�Cv)                                    Bxh�m$  �          @��\���
�x�ÿ���\)C{�{���
�hQ��G���Cz�)                                    Bxh�{�  �          @|(�?���n{?�@�C��\?���n�R������RC���                                    Bxhފp  �          @w
=?\)�s33>u@aG�C�1�?\)�o\)�5�)C�B�                                    Bxhޙ  �          @�Q�>�z��o\)�.{�$��C�>�>�z��\�Ϳ�=q��z�C�n                                    Bxhާ�  �          @�p����R���H>�@��RC�0����R��녿:�H�	��C�.                                    Bxh޶b  �          @�z��G���(�?��A`��C��;�G����ü��
���C��                                     Bxh��  �          @��>�������@Q�A���C���>������R?n{AG�C�l�                                    Bxh�Ӯ  �          @��?   ���@P��Bp�C��\?   ���R?��A�{C�~�                                    Bxh��T  �          @�>\���R@c�
B��C�L�>\��ff@G�A�z�C��q                                    Bxh���  �          @��>8Q���G�@g
=B	33C��
>8Q�����@�A���C��R                                    Bxh���  �          @ʏ\@��u@�{B3=qC�]q@����\@UB   C��f                                    Bxh�F  �          @��?����G�@z�HB$��C��?����(�@.�RA�C�,�                                    Bxh��  �          @�\)?�����@z�HB�RC��3?���
=@   A��HC�s3                                    Bxh�+�  �          @�  ?�p���G�@AG�A��C���?�p��˅?��HAK�C�xR                                    Bxh�:8  �          @�p�?�z���{@c33B  C���?�z���p�@(�A�  C��                                    Bxh�H�  �          @�=q?!G���Q�@�\A�C���?!G����?5@�
=C�^�                                    Bxh�W�  �          @��
?+���Q�@��A���C�˅?+����R?�ffA-p�C�}q                                    Bxh�f*  �          @�G�<�����@<(�A�Q�C�+�<����H?���AV{C�'�                                    Bxh�t�  �          @��
��33��Q�@&ffA��
C�/\��33���?�\)A)�C�T{                                    Bxh߃v  �          @��H?333���@_\)B�C�Ф?333��ff@�\A���C�\)                                    Bxhߒ  �          @�ff?c�
����@w
=B�RC�J=?c�
��33@ ��A�33C��f                                    Bxhߠ�  �          @�z�?��
��  @�p�B �C�}q?��
���R@@  A�  C�.                                    Bxh߯h  �          @�z�@!��z=q@��BD{C�q�@!�����@�{Bp�C�{                                    Bxh߾  �          @�  @�����@���B>ffC��f@�����\@�ffB�
C�\)                                    Bxh�̴  �          @�p�>u����@L��A�C�:�>u��?�Aq�C��                                    Bxh��Z  �          @��@#33��G�@�(�B8=qC��@#33��=q@��RB\)C�b�                                    Bxh��   �          A�
?�(����@�33B/�
C�u�?�(��ᙚ@�A�C�˅                                    Bxh���  T          Az�?W
=��{@��B��C��?W
=���@EA�Q�C��                                    Bxh�L  �          A��?�G���
@�Q�Bp�C�}q?�G���@=p�A�  C�
                                    Bxh��  �          A ��?�ff���@�  B��C�g�?�ff���@]p�A�G�C��
                                    Bxh�$�  �          AQ�@g
=�ָR@�33BC�&f@g
=� ��@��Aə�C��                                    Bxh�3>  �          A$��@H�����@�Q�B�C��@H�����@s33A�33C��
                                    Bxh�A�  �          A"�R@HQ��ᙚ@��
B"�C���@HQ��z�@��A�33C��                                    Bxh�P�  �          A"�R@K�����@�33Bz�C��=@K��
�H@�\)AĸRC��                                    Bxh�_0  T          A!�@G
=��(�@��
B�HC��@G
=�ff@k�A�z�C���                                    Bxh�m�  �          A!@AG���z�@���B�C�w
@AG����@P��A���C�7
                                    Bxh�||  �          A ��@$z��ff@��HA��HC�~�@$z���@{Adz�C���                                    Bxh��"  �          A��?���z�    ��C�H?���
=�   �qG�C�&f                                    Bxh���  T          A
=?����  ���� z�C�ٚ?�������y������C�%                                    Bxh�n  �          Aff?�
=���Y�����C�0�?�
=��\�XQ���{C�q�                                    Bxh�  �          A��?���(��u��\)C��{?������333��(�C���                                    Bxh�ź  �          A�?�\)��u��{C��?�\)�  �%�x(�C�9�                                    Bxh��`  �          A  ?˅�G���\�@  C��3?˅����A���
=C���                                    Bxh��  �          A�?�  �ff?
=@]p�C���?�  �\)���R�=p�C�                                    Bxh��  �          A��?��\��R?
=@\(�C��)?��\���   �=p�C��                                    Bxh� R  �          A�
?�ff���?=p�@�\)C�7
?�ff�{��33�/33C�L�                                    Bxh��  �          A33?�  �Q�?�\)@ϮC�]q?�  ���\���C�c�                                    Bxh��  �          AG�?�=q�ff?��
@�RC��=?�=q�=q������z�C��                                    Bxh�,D  �          A\)?�z���
?�33A��C�}q?�z��p��s33���C�t{                                    Bxh�:�  �          A��?������?��A�C�1�?����
=�n{���\C�'�                                    Bxh�I�  T          Aff?����?�(�A#�
C��?��\)�\(����C��                                    Bxh�X6  �          A
=?���  @�
AE�C���?������\�B�\C��)                                    Bxh�f�  �          A33?�=q���@�AA��C��{?�=q��
����P��C�u�                                    Bxh�u�  �          Az�?��R��H?У�A�C��{?��R�Q�xQ���\)C��f                                    Bxh�(  �          AQ�?^�R���@�z�A�ffC��?^�R���?���A�\C��3                                    Bxh��  �          A\)?=p��Q�@�=qA��HC�|)?=p��
=@AF�RC�>�                                    Bxh�t  �          A��?W
=�=q@�  A�G�C���?W
=�
=@&ffA{\)C���                                    Bxh�  �          A��?�p�� z�@��A�z�C�W
?�p��p�@'�A~�RC��R                                    Bxh��  �          AQ�?�  �G�@��A�(�C��?�  �33?�A:�\C��                                     Bxh��f  �          A�?��
�	�@l��A�=qC��R?��
���?���@�
=C�l�                                    Bxh��  T          A
=?���
=q@S�
A�Q�C�%?����?Q�@�\)C��                                    Bxh��  �          A�H?�
=��
@4z�A�  C�8R?�
=��H>��
?�
=C��                                    Bxh��X  �          A��?�z��z�@]p�A�=qC��?�z��=q?k�@�Q�C�B�                                    Bxh��  �          A�R?�
=�	��@Mp�A�  C�U�?�
=��\?8Q�@��
C��
                                    Bxh��  �          A��@
=���@C�
A���C�� @
=���?�@FffC�ff                                    Bxh�%J  �          Az�?�z��Q�@Tz�A�33C�Z�?�z��p�?B�\@��C��                                    Bxh�3�  �          A
=?�\)�	��@\(�A���C�T{?�\)��?k�@��
C���                                    Bxh�B�  �          A��@p���@�z�B{C�W
@p���
@8��A��C�Ff                                    Bxh�Q<  �          A�@Q���\)@���B��C�@Q��@s33A���C�XR                                    Bxh�_�  �          A��@j�H��
=@���B3  C���@j�H��\@�  A�=qC���                                    Bxh�n�  �          A(�@e���@�p�B2(�C��@e��(�@���A�\C��)                                    Bxh�}.  �          AQ�@j�H����@�p�B:�C��@j�H���
@�z�B��C��H                                    Bxh��  �          A=q@J=q�	@?\)A��\C��@J=q���>�@.�RC��\                                    Bxh�z  T          A  @E���@]p�A���C��=@E��H?k�@�{C�H�                                    Bxh�   �          A(�@H���33@mp�A���C�33@H����\?�
=@�{C�y�                                    Bxh��  �          A��@l���	�@>{A�Q�C���@l���>�
=@�HC�\                                    Bxh��l  �          Ap�@g��=q@�A�33C�  @g���
?��HA z�C���                                    Bxh��  �          A   @u���p�@�33A�  C�E@u����@.{A}p�C�                                    Bxh��  T          A�
@L���p�@��A�\)C���@L���{@z�AW�C���                                    Bxh��^  �          A�@{��(�@�z�B(�C��
@{��
@^�RA�C��                                    Bxh�  �          A�\@$z����@�ffBC�E@$z���@Q�A�
=C���                                    Bxh��  �          A�R@!���(�@���B�HC�/\@!��\)@W
=A���C��                                    Bxh�P  �          Aff@Dz�����@���B C��f@Dz��\)@4z�A��RC�u�                                    Bxh�,�  �          A�
@,����{@�p�BF�
C���@,����=q@�\)B	33C��                                    Bxh�;�  �          A��@ ����Q�@�\B9��C��@ ��� Q�@��A�p�C��                                    Bxh�JB  �          Aff@6ff��Q�@�  B1�C�C�@6ff��@�p�A��
C��                                    Bxh�X�  �          A33@W�����@�{B�
C�o\@W��ff@�Q�A�C�:�                                    Bxh�g�  �          A�@8�����
@��RBQ�C�e@8����@A�A�  C��                                    Bxh�v4  �          AG�@Z�H�Q�@���A��C�:�@Z�H�G�?��HA��C�Q�                                    Bxh��  �          A(�@G���@�(�A���C��3@G���
?�{A0z�C��3                                    Bxh㓀  �          A�R@i���ָR@�(�BG�C�H�@i���{@k�A�p�C��                                    Bxh�&  �          A�@[����@�  B2�C��@[���@��A�RC��                                    Bxh��  T          AQ�@�=q��33@\B��C�  @�=q���H@~{A�\)C�k�                                    Bxh�r  �          A�@�z���ff@�B-�HC��q@�z�����@�
=A��HC���                                    Bxh��  �          A��@x����Q�@�Q�B4G�C��q@x����@�33A�Q�C�q                                    Bxh�ܾ  �          A�@b�\��  @�RBB�C��q@b�\��z�@�33B�C�33                                    Bxh��d  �          A�R@n{��  @��B3\)C�t{@n{��  @�=qAC�33                                    Bxh��
  �          A��@O\)���@�
=B:�C��
@O\)����@�
=A�  C��                                     Bxh��  �          A=q@u��Q�@���BP�
C��@u��Q�@�=qB�C�Ǯ                                    Bxh�V  T          A�R@����g�@��HBVG�C�@�����(�@�{B"��C�˅                                    Bxh�%�  �          A
=@����\)@�Be��C��@����G�@�Q�B'
=C�t{                                    Bxh�4�  �          A�
@Mp���33@��RB*�HC��=@Mp����
@���A�z�C�%                                    Bxh�CH  �          A��@b�\?��\@�33B��)A��@b�\��\)@��
B��C�0�                                    Bxh�Q�  �          AQ�@�  ��(�@�
=BrG�C�ff@�  �|(�@ۅBJC��q                                    Bxh�`�            A@}p���A(�B|  C�'�@}p��tz�@�RBV\)C��                                    Bxh�o:  �          A  @|�Ϳ���A�B|G�C���@|���p  @�ffBW33C�B�                                    Bxh�}�  �          AQ�@}p����A ��Bq��C�*=@}p���=q@�Q�BD�HC�xR                                    Bxh䌆  �          A�H@l�����\@�\BA�C��@l����
=@�\)B
=C��q                                    Bxh�,  �          A��@J=q���H@�  BQ��C��3@J=q���
@�ffB
=C�Y�                                    Bxh��  �          A@%��Q�@�
=BO�\C�"�@%��Q�@�G�B��C��3                                    Bxh�x  �          A�H@Z�H��G�@�\B=��C���@Z�H��p�@�=qA��HC�]q                                    Bxh��  �          A�H@z�H���@�=qBffC���@z�H�p�@;�A��C���                                    Bxh���  �          Aff@s33��(�@�  B�C��@s33�=q@5A�G�C�.                                    Bxh��j  �          Aff@�����ff@@  A�33C��@�����
>�G�@(��C�R                                    Bxh��  �          AG�@U���H@�33A�p�C��f@U�\)?\A�C�xR                                    Bxh��  �          A�@=q��33@θRB*ffC�
=@=q�  @�z�AͮC�%                                    Bxh�\  �          A�\@~{����@�G�B+C�/\@~{���@�ffA��C��                                    Bxh�  �          A�@����@�
=B0\)C���@���߮@��A���C���                                    Bxh�-�  �          AQ�@������@�(�B/G�C�c�@����ff@���A�
=C���                                    Bxh�<N  �          A��@��\���
@���B2�C�%@��\���R@�=qBz�C��                                    Bxh�J�  �          A(�@X����  @�  BP��C�L�@X����ff@�33B�C�XR                                    Bxh�Y�  �          A��@�����(�@\B(��C�l�@����߮@��
Aأ�C���                                    Bxh�h@  �          A{@�p���G�@�  B(��C��@�p���
=@��Aܣ�C�4{                                    Bxh�v�  �          A@�z���(�@�(�A��
C�Ff@�z�����@�\AK
=C���                                    Bxh兌  �          A(�@}p���z�@�  A�(�C�� @}p����?�=qA7
=C��                                    Bxh�2  �          A  @:�H��  @��
A��C�� @:�H��
?���A733C�|)                                    Bxh��  �          AQ�@fff���
?�A3�C��H@fff���׿J=q���C�n                                    Bxh�~  �          A\)@�G�����@�B{C�J=@�G���33@G�A���C���                                    Bxh��$  �          A�\@��H���@�  B=qC�p�@��H��R@I��A���C��                                    Bxh���  �          A�@xQ����@ȣ�B5�RC�q�@xQ���G�@�ffA���C�Y�                                    Bxh��p  �          A�@mp����H@���BI�C�C�@mp���G�@�{B�C�Ff                                    Bxh��  �          AG�@>{��\)@�
=Bp�C��=@>{� z�@5A���C�'�                                    Bxh���  �          A	@
=?�ff@�  B��B
=@
=�xQ�@�33B��fC���                                    Bxh�	b  �          Az�@33?�Q�AB���A�z�@33��Q�A�B��)C��q                                    Bxh�  �          A
=@+�>\)@��RB��R@;�@+����@��HB�\)C�%                                    Bxh�&�  �          A��@S33�5@��B���C��
@S33�HQ�@��
Be�
C�C�                                    Bxh�5T  �          A�@��\�Tz�@�(�Bu�C�G�@��\�Mp�@�
=BUQ�C��=                                    Bxh�C�  �          A{@�녿�(�@�Bd�
C�\@���p��@ҏ\B@=qC�<)                                    Bxh�R�  �          A��@��\�@  A   Bzz�C�Ф@��\�P��@�33BZffC���                                    Bxh�aF  �          A@�=q��(�@��Bq�RC�{@�=q�l(�@�33BMffC��q                                    Bxh�o�  �          A�H@���=p�@��RBq  C�N@���P  @陚BS33C��                                    Bxh�~�  T          Az�@�
=�k�@�BU
=C�5�@�
=�P��@׮B:�\C���                                    Bxh�8  �          A  @�
=�}p�@�BA�C�t{@�
=�N�R@�p�B)�C�Ff                                    Bxh��  �          A{@�z�.{@��B@=qC���@�z��<��@��B+��C��)                                    Bxh檄  �          A�H@�=q��
=@�z�B,�C��@�=q�x��@�B�\C���                                    Bxh�*  �          A�\@�zὸQ�@�\)B9��C��R@�z��ff@�=qB,33C�B�                                    Bxh���  �          AG�@�\)�,(�@�=qB�C��@�\)��@�(�A���C�Q�                                    Bxh��v  �          A\)@�p��c33@��RB
33C���@�p�����@qG�A��C�n                                    Bxh��  �          Az�@�{�!G�@��HBKp�C��@�{��{@�(�B =qC���                                    Bxh���  �          A��@�G����@�(�BffC��@�G���@�A�(�C�+�                                    Bxh�h  �          A�@�33���@�BC�(�@�33��
=@O\)A�ffC���                                    Bxh�  �          Az�@������@��\A�C�3@�����@5A��RC���                                    Bxh��  �          A�H@�p�����@�
=A�z�C��
@�p���\)@;�A���C��\                                    Bxh�.Z  �          A�H@�p����\@���A��HC��@�p���
=@z�AdQ�C�J=                                    Bxh�=   �          A�H@����G�@��\A���C�xR@���ҏ\@4z�A��C�W
                                    Bxh�K�  �          A=q@��R����@��A�ffC�@��R���?��A=�C�Ǯ                                    Bxh�ZL  �          A��@�����@l��A�G�C��R@�����
?�G�AffC���                                    Bxh�h�  �          A��@�{���R@o\)A���C�p�@�{��(�?��A3\)C�\)                                    Bxh�w�  �          A	�@�  ���@u�A��
C���@�  ��\)?��HAT��C��                                    Bxh�>  �          A�
@��\���
@UA�ffC�XR@��\��{?���A{C�9�                                    Bxh��  �          AG�@�Q���=q@r�\A�ffC���@�Q���=q@33Af=qC��R                                    Bxh磊  �          Ap�@\)����@��HB*C���@\)���@p  A�ffC�z�                                    Bxh�0  �          A��@Å����@O\)A�33C�XR@Å��
=?�  A&{C�3                                    Bxh���  �          A@�p��Z=q@��RB%�C�N@�p���G�@�  A��
C��q                                    Bxh��|  �          Ap�@����Q�@r�\A��C���@������@��As
=C���                                    Bxh��"  �          A��@�Q��S33@j�HA�p�C��3@�Q����H@ffA��C�*=                                    Bxh���  �          A  @ȣ���  ?�{ATQ�C�l�@ȣ���33>��?��
C�q�                                    Bxh��n  
�          AQ�@Ϯ��Q�?�Q�A>=qC��q@Ϯ�����#�
���
C��                                    Bxh�
  �          AQ�@љ����@�=qB	��C�S3@љ��1�@w
=A�C�|)                                    Bxh��  �          A�\@��?�
=@��RB)ffA��\@����=q@�ffB3
=C���                                    Bxh�'`  �          A�@���>\@�Q�BS�H@��R@��Ϳ���@�Q�BHG�C��                                    Bxh�6  �          A�@�  �=p�@�G�B=��C�0�@�  �0  @�B%C��)                                    Bxh�D�  �          A(�@�z�Ǯ@�Q�BE��C���@�z��fff@��B!��C�y�                                    Bxh�SR  �          A�@����Q�@�(�B5z�C��@������@�
=B(�C��                                    Bxh�a�  �          AG�@��
��R@��BV�\C�]q@��
�7
=@���B;�HC���                                    Bxh�p�  �          A(�@�z�}p�@ʏ\BIC���@�z��Fff@��B,Q�C�t{                                    Bxh�D  �          Ap�@���>�z�@ӅBU��@Vff@����   @��BH  C���                                    Bxh��  �          A
=q@�{?��H@���BOG�AQG�@�{����@���BO\)C�}q                                    Bxh蜐  �          A�@�z�>��H@�(�BF
=@��R@�z��=q@���B<��C�f                                    Bxh�6  �          A
�R@���=�\)@���B6(�?�R@�����
@�G�B)  C���                                    Bxh��  T          A��@�녿��@��
B  C���@���E@��HA�{C�˅                                    Bxh�Ȃ  �          A�@�\��(�@�\)A���C�S3@�\�;�@l(�Aȏ\C�
                                    Bxh��(  �          A�
@��
�Ǯ@��A�RC��@��
�@  @eA�ffC��                                    Bxh���  �          AQ�@�p���33@�ffA�(�C��f@�p��7
=@k�A��HC�q�                                    Bxh��t  �          A	�@�{�n{@���B�C�N@�{�\)@���A�G�C�n                                    Bxh�  �          A�H@��>�z�@��RB7(�@S�
@�녿�ff@��B-(�C��H                                    Bxh��  �          Az�@�{?=p�@�  B�@��H@�{�xQ�@��RB�C��                                    Bxh� f  �          AG�@�?�Q�@�z�B5�
A<��@��u@�{B7�C�9�                                    Bxh�/  �          A�
@S�
���@�z�B���C�� @S�
�l��@أ�BV�HC��                                    Bxh�=�  �          A	�@C�
����@���B���C�H�@C�
�y��@�
=BZ=qC��                                    Bxh�LX  �          A�
@C�
����A��B�W
C�U�@C�
�w
=@�ffB^z�C�4{                                    Bxh�Z�  �          A	@$zᾸQ�@��B�{C�
=@$z��J=q@�33Bt�C���                                    Bxh�i�  T          A
ff@�녿��H@�G�BGz�C�Q�@���k�@�33B#��C���                                    Bxh�xJ  
�          A	@�z��\)@�G�B3z�C�*=@�z����@��B
�C���                                    Bxh��  �          Az�@��\��p�@��HB�
C��@��\����@[�A�33C�Y�                                    Bxh镖  �          A�@c�
�Vff@��
BV33C�^�@c�
���@��Bp�C�}q                                    Bxh�<  �          A	G�@^�R��33@���B|�C��@^�R�|��@�p�BJ��C��{                                    Bxh��  �          A��@~{��@�=qB'  C��q@~{���@eA��C��                                    Bxh���  �          A=q@n�R�У�@�{B 
=C���@n�R��Q�@33AV=qC��{                                    Bxh��.  �          A��@tz���(�@�ffBp�C�~�@tz�����@Mp�A���C�p�                                    Bxh���  �          A��@z�H��p�@���B�HC���@z�H����@Q�At  C�Z�                                    Bxh��z  �          A  @#�
�J�H@ʏ\Bd�C�}q@#�
��\)@���B(�C��                                    Bxh��   �          A
{>\�p��A�B�Q�C��>\�{�@��By��C��)                                    Bxh�
�  �          A{?333��{@��B^  C��?333��\)@��B	�C��f                                    Bxh�l  �          A�?����\@ҏ\BAz�C��q?����@���A�C��{                                    Bxh�(  T          Aff?У���=q@��B*G�C�>�?У�� ��@S33A���C���                                    Bxh�6�  T          A�?\�Ӆ@�B�C�z�?\��@9��A���C�=q                                    Bxh�E^  �          A��?������
@�z�B{C��3?����@!G�A�z�C��=                                    Bxh�T  �          AG�?\��G�@��
B��C�{?\��R@p�Ag�C��                                    Bxh�b�  �          A{?�\)��R@��A�Q�C��?�\)���?�=qA
�HC�,�                                    Bxh�qP  �          A{@33� ��@8Q�A��C���@33��;���ٙ�C��                                    Bxh��  �          A
=@33��ff@g
=A�\)C�S3@33��
?
=q@^�RC��R                                    Bxhꎜ  �          A
=?�G���?�R@\)C���?�G��p��'����HC���                                    Bxh�B  �          A(�?�(���
��ff��{C��?�(���Q����H��G�C���                                    Bxh��  �          A
=?�ff�
=���Dz�C�]q?�ff��ff��ff��
C�L�                                    Bxh꺎  �          A��?�����H?�@h��C���?��������!�����C���                                    Bxh��4  �          A\)@Tz��˅@��HBp�C�Ǯ@Tz���?�AO�
C���                                    Bxh���  �          A ��?�=q@?\)@�Q�B��\Bx��?�=q>8Q�@��
B�u�@�{                                    Bxh��  �          A{>�(�@c�
@��B�{B�G�>�(�?�\A��B���BH�\                                    Bxh��&  T          A�H?��?@  A��B�B��?����
@���B�  C�
                                    Bxh��  �          A>\@@�B���B��>\��G�A=qB��fC�C�                                    Bxh�r  �          A�@,�ͽ�@�{B�k�C�@,���Dz�@�\Bs\)C���                                    Bxh�!  �          A�H>���?�
=@���B��B��\>��Ϳ�z�A ��B��C���                                    Bxh�/�  �          A�?�ff?�z�Az�B�.BQ�?�ff���A\)B���C���                                    Bxh�>d  �          A�@7
=?�
=@�ffB�.A��@7
=��z�@���B���C��{                                    Bxh�M
  �          A��@z�>�Q�A�\B��fA��@z��5�@�B�G�C�#�                                    Bxh�[�  �          A�R@TzὸQ�A�
B��C�0�@Tz��L(�@�33Bk=qC�{                                    Bxh�jV  �          A
=q@W
=?�  @�G�B��{A��H@W
=���
@�B~�C��                                    Bxh�x�  �          A@�
=��{@��Bp
=C���@�
=�xQ�@ҏ\BC�RC���                                    Bxh뇢  �          Az�@q녿�=qA z�BC�@q���G�@��BO=qC���                                    Bxh�H  �          AQ�@\)?uAp�B��=A�33@\)��@�33B���C�޸                                    Bxh��  �          A  >��@QG�@��B��\B���>��>��@�
=B���Bs�                                    Bxh볔  �          A�\�a�@�G�@w�A癚B��a�@��
@�(�B=33C�)                                    Bxh��:  �          AG���G�@1�@�p�B��B��)��G���{AffB��RCC@                                     Bxh���  �          A
=q�?\)@���@���B<
=B�k��?\)@��@�\)B�B�C�3                                    Bxh�߆  �          A  ��p�@�
=@��Bp�C����p�@8��@�p�BJ33C:�                                    Bxh��,  �          A
=���R@�z�@��
B,��Cz����R?�ff@�33B_��C��                                    Bxh���  �          A(����
@�@�Q�BDffC5����
���R@�p�BU��C7z�                                    Bxh�x  �          A�����R?@  @�G�BqC)�����R���@�  Bc{CO��                                    Bxh�  �          A(���=q>L��@�Q�Be33C1�H��=q�-p�@�Q�BO  CQQ�                                    Bxh�(�  �          A\)�tz�xQ�@�
=B~�CBJ=�tz��~{@�\)BN�
Cb
                                    Bxh�7j  �          Az�Y���P��A�RB�.Cff�Y�����@���B5{C�3                                    Bxh�F  T          A�Ϳ�z��1G�A
�RB��Co
=��z�����@�p�BA��C~G�                                    Bxh�T�  �          A�׿
=q�Y��A	��B�Q�C��׿
=q��
=@ӅB6{C��                                     Bxh�c\  �          A�R�G����A=qB�k�C>  �G��j�H@�G�Bb33Ce�\                                    Bxh�r  �          AG������A�B�u�Cb�3�����G�@��RBd�C~ff                                    Bxh쀨  �          A��L����\Ap�B�W
Cz��L����G�@�  BQ=qC��f                                    Bxh�N  �          A녾L���(��A
�HB�
=C��þL����=q@޸RBHffC��                                    Bxh��  �          A녿������RA
=qB�#�Ci𤿹�����R@�BT  C~n                                    Bxh쬚  �          A�׿L�Ϳ�=qA�B���Cn�R�L����G�@��
Bi=qC�@                                     Bxh�@  �          AQ��>�AQ�B�aHC/����S�
A�B��)CsG�                                    Bxh���  �          A
�\�c�
�#�
A{B��\C4���c�
�L��@�
=B�{C~��                                    Bxh�،  �          A
{�\)��HA ��B�C��f�\)��z�@θRBH  C�                                    Bxh��2  �          A  ��33��
=@ۅBJ33C�f��33��
=@���A���C�޸                                    Bxh���  �          A(�?O\)�Q�A	G�B�ǮC�G�?O\)��z�@�=qBs�C��H                                    Bxh�~  �          A
=?�����A�B���C�� ?�����R@�BX�C���                                    Bxh�$  
�          A
=?���s33AQ�B�#�C��?������A=qBrp�C��                                    Bxh�!�  �          A��?�  �L��A�RB��{C���?�  ��
=Ap�Bt�C�G�                                    Bxh�0p  �          A\)@]p�=L��A
=B��?^�R@]p��Y��@��Bj��C��                                    Bxh�?  �          A�@X���!�@�(�Bw�C���@X����ff@�
=B0p�C��=                                    Bxh�M�  �          Az�@G
=��  @�(�BU�C�T{@G
=��=q@�
=B�C��                                    Bxh�\b  �          Aff@�ff��  @��B8�\C��=@�ff��@��\AڸRC��                                    Bxh�k  |          A�@�\)��p�@�G�A�\)C���@�\)�B�\@y��A�ffC���                                    Bxh�y�  �          A��@�33���H@�=qBZffC��R@�33��33@�  B)G�C��)                                    Bxh�T  �          Aff@�{���@�z�BXffC��@�{���@ǮB,��C��                                     Bxh��  �          A�R@�Q쿾�R@�p�Bc�C�P�@�Q���{@�B2��C�|)                                    Bxh���  �          A��@~{>�
=Az�B��)@�\)@~{�C33@�\)Bg��C�<)                                    Bxh��F  �          A33@w
=?�Q�A�RB}�RA��@w
=��A{B{��C�8R                                    Bxh���  T          A�\@n{>.{AQ�B��q@*=q@n{�Q�@��Bg
=C�S3                                    Bxh�ђ  T          Ap�@
�H<#�
A(�B�#�>�=q@
�H�a�A Q�Bz�C�Ф                                    Bxh��8  �          A�@��?n{A(�B�ffA�G�@���\)@�p�B��)C��{                                    Bxh���  �          A��@&ff��=qA
=B��qC�l�@&ff��G�@�ffBT�C��                                    Bxh���  �          A�@Mp��p�@�33B{�C�G�@Mp���@�B1�C�G�                                    Bxh�*  �          A�H@*�H?�A�B�k�B
��@*�H�ٙ�A��B��=C��                                    Bxh��  �          A@z�>W
=A
�HB���@��\@z��UA (�B|G�C�\)                                    Bxh�)v  �          Ap�?����6ffAz�B��C�:�?����\@�(�B8G�C���                                    Bxh�8  �          A@����=q@���BV�C��f@����  @��A�
=C��f                                    Bxh�F�  �          Az�@%��ff@�p�B*�
C��@%�(�@N{A�
=C�w
                                    Bxh�Uh  �          Ap�@Q��Å@�p�B+(�C��@Q��33@VffA�Q�C��                                    Bxh�d  �          A
=@����R@��HB`��C�*=@���Q�@�
=BffC��q                                    Bxh�r�  �          A
=?�33�s�
@�\)B|�C�\?�33��  @��RBz�C��R                                    Bxh�Z  �          A��@z����@�z�B^�
C�>�@z���z�@�G�BG�C���                                    Bxh�   
�          A
=@e��\)@�G�B?z�C��q@e���H@�(�AָRC�                                    Bxh  �          A�H@E���  @���BFffC���@E����@�
=A�{C�H�                                    Bxh�L  �          A�\@s33?L��@��HB~�RA=�@s33� ��@�
=Bj�\C�>�                                    Bxh��  �          A�@fff���@���B}G�C��\@fff��(�@��BA�C��{                                    Bxh�ʘ  �          @���@\���Q�@���BeG�C��{@\�����
@��RB�
C��f                                    Bxh��>  �          Az�@�{>��@��HBh��@�  @�{�.{@�33BR�\C��\                                    Bxh���  �          A
=@�ff��
=@�Q�B_z�C�J=@�ff��ff@��HB1G�C��)                                    Bxh���  �          A�\@qG��L��A33B���C��H@qG��e@�(�B[�C�+�                                    Bxh�0  �          A�H@�(���ff@�z�Be��C��H@�(����H@ə�B0�C���                                    Bxh��  �          A�@��R��z�@�ffBbG�C���@��R��ff@�ffB(�C��                                    Bxh�"|  �          A@�z��:�H@�B\G�C�1�@�z����@�z�B  C���                                    Bxh�1"  �          A�H@�Q���H@���Ba��C�˅@�Q���G�@ǮB'��C��\                                    Bxh�?�  �          A�
@���   @���B[�HC�q@����\)@���B�\C�s3                                    Bxh�Nn  �          A(�@��H>B�\A ��Bp�@ff@��H�L��@��
BS�C��                                    Bxh�]  �          A�
@��׾�z�@���B]p�C�h�@����^�R@��B<��C�H�                                    Bxh�k�  �          Aff@��Ϳ   @�{B`ffC�8R@����l(�@أ�B;�C�(�                                    Bxh�z`  �          A��@�����G�@��\B`(�C��3@�����p�@θRB,�
C���                                    Bxh�  �          A��@�=�@��Bbz�?��@��Mp�@�BF�C�R                                    Bxh  �          A�
@У׿�33@�z�B3�C��@У����\@�p�B��C��q                                    Bxh�R  �          A@�Q�#�
@���B1�\C�H�@�Q��Y��@�  B�C���                                    Bxh��  �          AQ�@��H�!G�@�
=B1Q�C�J=@��H�S�
@��HB��C��=                                    Bxh�Þ  �          A
�H@���(�@���BS33C��)@���[�@�(�B.C��                                    Bxh��D  �          Az�@p  >�@�(�B{�@�{@p  �.{@��
B_��C�                                      Bxh���  �          A��@}p�?�ff@�Bv  A�p�@}p��ff@�  Bl��C��                                    Bxh��  �          A�
@u�    @��HB�HC��)@u��Vff@��HBYffC�k�                                    Bxh��6  T          Ap�@��\�8Q�@�33Bk��C�ٚ@��\�Z=q@�G�BGC���                                    Bxh��  
�          A�@��?5@��Bn=qA
=@���%�@�33BY�C��{                                    Bxh��  �          A��@�?0��@�\)BZ
=@�
=@���@�33BIz�C�,�                                    Bxh�*(  �          A��@�=q���
@��B`Q�C�� @�=q�I��@�{B@C�h�                                    Bxh�8�  �          A�R@��>�@���BUQ�@���@���!�@�{BA�
C�ٚ                                    Bxh�Gt  �          A�@��׿��@�RBp  C��q@����p  @�\)BB��C�T{                                    Bxh�V  �          A�@���@@��BT�A�  @��ÿz�H@��B`�\C�޸                                    Bxh�d�  �          A@�z�?��@��B@�\A/33@�z���
@�Q�B:ffC�<)                                    Bxh�sf  �          A�H@���?h��@��
BJ��A��@����	��@ҏ\B?��C�j=                                    Bxh��  �          A  @�\)�5@ǮB9�C�� @�\)��Q�@���A�p�C�j=                                    Bxh�  ,          Ah  @�
=���A&�\BCz�C���@�
=�-�@�=qA��C��H                                    Bxh�X  �          A��@�G����Af�HBS(�C��{@�G��j{A�A�\)C��=                                    Bxh��  �          A�G�@�33�"�HAk
=BN�C�ff@�33�x��A��A��\C��                                    Bxh�  �          A�@���HAmBSC�1�@��r�HA�RA�{C�AH                                    Bxh��J  T          A���@�{���
Ad��BW(�C���@�{�JffAz�B�HC��                                    Bxh���  �          A�(�A
=��=qAP  BI�C�aHA
=�*�HAQ�B\)C�Y�                                    Bxh��  �          A��A���HA?\)B,�\C��A��W�@�Q�A�C���                                    Bxh��<  �          A���A#�
�,Q�A{BC�ǮA#�
�^=q@|��ANffC�1�                                    Bxh��  �          A��\A4z��733@�(�A��\C�H�A4z��E��Y���:=qC�<)                                    Bxh��  �          A�{A_�
��
@w�AN=qC��A_�
�#��.{�  C��                                    Bxh�#.  �          A��AP(��Z�R@�A���C�˅AP(��o�
��ff���C�y�                                    Bxh�1�  �          A�ff@����@��\A}C�@�����=p���C��q                                    Bxh�@z  �          A��RA4z��F=qA (�A�ffC�(�A4z��jff?޸R@�(�C���                                    Bxh�O   �          A��HAH  �_
=@�=qA~{C��\AH  �m��{��Q�C��                                    Bxh�]�  �          A��
AJ{�O\)@�Q�A��C�"�AJ{�k�?z�?��HC�O\                                    Bxh�ll  �          A�G�Aap��4z�@�A�(�C��=Aap��T(�?�33@���C�^�                                    Bxh�{  �          A�G�A9���g�@�z�A��C�Y�A9���u녿�����p�C��                                    Bxh�  �          A��R@|�����\@{@��C�{@|����(�������\)C�Y�                                    Bxh�^  �          A�@��R����@AG�A��C��)@��R��33�����  C�8R                                    Bxh�  �          A�p�@����z�?�(�@��\C��@����  ��
=���C��H                                    Bxh�  �          A���@\(����H@z�@�C��R@\(����
�������C���                                    Bxh��P  �          A�{@�ff����?s33@;�C���@�ff��  ��z���{C�%                                    Bxh���  �          A��H@�=q��p��\(��.�RC���@�=q�h���33��p�C�%                                    Bxh��  �          A�33@�(��lz��g��H  C�z�@�(��9��G��=qC�Q�                                    Bxh��B  �          A�  @�{�j{������RC�q�@�{�-�0  �,�C��                                    Bxh���  |          A�33A{�O�@G�@�{C���A{�J{�qG��S�C�f                                    Bxh��  �          A��A]G��{@�\)A��
C��fA]G��(  ?�33@r�\C�c�                                    Bxh�4  �          A��
Av�R�^�R@9��A&{C���Av�R���?��
@j�HC��                                    Bxh�*�  �          A�z�AY��z�@�z�A�ffC���AY�� ��>�(�?��HC��\                                    Bxh�9�  h          A�33@�����  ��33�[33C���@����`(��C33��C���                                    Bxh�H&  �          A�G�@�\)���@��HAe�C�3@�\)��p��h���5C���                                    Bxh�V�  �          A�z�@�������@�
=Ak
=C��@�����33�`  �-��C��=                                    Bxh�er  �          A�  @%�����@'
=A33C��@%�������Q����RC�B�                                    Bxh�t  �          A�@��
����*�H�(�C�E@��
�[
=��R�Q�C���                                    Bxh�  �          A�Q�A�p���?\)�
=C�� A�@���\)��=qC���                                    Bxh�d            A�33A#33�9�����
C��fA#33�Å�Y�Cp�C��=                                    Bxh�
  �          A��A!�����R�QG��@�C���A!��?�R�e�[\)@_\)                                    Bxh�  �          A���A�
@�  �ap��aQ�A�  A�
A\)�+�
�z�BF33                                    Bxh�V  �          A���AI���(���H��RC�]qAI�� ���=��+=qC�W
                                    Bxh���  �          A�A�
�e�����vffC��RA�
�*=q�,(���C��)                                    Bxh�ڢ  �          A�  A=q�m������C�.A=q�E���=q���C��q                                    Bxh��H  
�          A���@˅�Y������{C��f@˅� ���#��"C�*=                                    Bxh���  ,          A��R@�Q������H���C��@�Q�����6�H�l�C�'�                                    Bxh��  g          A�33@�G��/
=A$(�B!�RC�\)@�G��f�R@�(�Ar{C��                                     Bxh�:  �          A���@���p�>�G�@�C�z�@���(��xQ���z�C�˅                                    Bxh�#�  |          A���@���dQ������ߙ�C�Ff@����H�Up��[  C��3                                    Bxh�2�  h          A�G�Aff��{�?\)�2C�|)Aff�0���\���Y33C�                                    Bxh�A,  T          A��\Az���Q��H���I  C��RAz�?�p��Q���U��A;�                                    Bxh�O�  �          Ahz�@0  ��
=�XQ�\C�S3@0  @���H��{Bw�H                                    Bxh�^x  h          A]��@E�@'
=�P(�z�B!(�@E�A���%���EB�B�                                    Bxh�m  �          A]p�?=p�?ٙ��X��¤aHB�\?=p�@�  �3��](�B�{                                    Bxh�{�  �          A`���g
=?�  �R�\�Ch��g
=@����/�
�T�B���                                    Bxh�j  	�          Ag�?�G���{�]¡�{C��?�G�@�  �R�\
=B�.                                    Bxh�  p          A�33@,(����\�t  C��3@,(�@L���xz���BG                                   Bxh�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxh�\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxh��  R          A�\)>��
�\(��=���"�C��=>��
��ff��
=��C�]q                                    Bxh�Ө  
z          A�{@X���J=q�B=q�+z�C�� @X���������8RC���                                    Bxh��N  
�          A��
?O\)�M���E�/\)C��?O\)��z���G�ǮC�R                                    Bxh���  
2          A�p�@�H�G
=�Ap��.��C���@�H��z�����=C��                                    Bxh���  
n          A��R@(���\�x(��u�C��H@(��#�
��(�¥C��                                    Bxh�@  �          A�A{�G33�������C���A{�\)�2{�.  C��                                    Bxh��  	�          A�=q@�G���(��o33\C��3@�G�@*=q�x  W
B=q                                    Bxh�+�            A�@�  �Jff�8  � 
=C�C�@�  ��=q���\C��                                    Bxh�:2  
�          A��\@��
��������G�C�{@��
�G��Qp��.�C�"�                                    Bxh�H�  �          A�Q�@��������s�
�<(�C��{@����\Q��9���p�C�H                                    Bxh�W~  �          A��@�(����R?���@�z�C�,�@�(����H�������C���                                    Bxh�f$  �          A�G�?�{���\@�{A�{C��f?�{�����Z=q�5�C��
                                    Bxh�t�            A��@.�R����?��@��C���@.�R�|  ��\)��ffC��=                                    Bxh�p  �          A���@�R���@#�
A�
C�&f@�R������
=��C�XR                                    Bxh��  �          A�z�?�G��j�H@��HA���C�u�?�G���ff��=q�uC�@                                     Bxh���  h          A���@Dz���
=>�  ?\(�C�aH@Dz��e����(���p�C��                                    Bxh��b  "          As33@j=q�Y����33��(�C���@j=q�\)�+
=�7ffC�T{                                    Bxh��  �          Ap  @`���U���\��\)C�]q@`�����!��1p�C��=                                    Bxh�̮  �          AuG�@U�b=q@��A�C���@U�f�H�@���7�C��f                                    Bxh��T  �          Ax��@�=q�i�@N�RA@��C��@�=q�e���G��r{C���                                    Bxh���  �          A�\)@���u��z���z�C���@���J{�33��
C���                                    Bxh���  
�          A��@��H�t���l(��A��C���@��H�;��'�
�z�C��                                     Bxh�F  �          A�(�A Q��F�H��p�����C�k�A Q������8���6��C��
                                    Bxh��  r          Av�\A	G��p��z��{C��A	G��K��;
=�OC��
                                    Bxh�$�  
�          Ac�
@��ÿ�=q�7�
�`C��@���@�=q�-�O
=A���                                    Bxh�38  �          At��@��R>����Q��m@�R@��R@�\)�5�AB {                                    Bxh�A�  �          A8Q�@5��G���G��$  C�s3@5��������RC��f                                    Bxh�P�  
�          A9�?�\)�((�@xQ�A��
C��?�\)�1p��˅� ��C���                                    Bxh�_*  �          A9�?�p�� zᾔz��33C�33?�p��
�H��G���(�C��R                                    Bxh�m�  �          Ae�?c�
�]�@Q�A�C��R?c�
�UG��������C��                                    Bxh�|v  �          A`Q쾨���]����Q���C�O\�����7�
��(��	�RC�+�                                    Bxh��  �          Abff@C33�W
=�\)�#�C�b�@C33�)�	����C�H                                    Bxh���  �          Af=q@0���D  ��=q����C�Z�@0�����
�:{�\z�C��R                                    Bxh��h  �          Ag
=@a��4����H�
�C��=@a���Q��I���t(�C�4{                                    Bxh��  �          Alz�@_\)�.�H�\)�  C��)@_\)���U��qC���                                    Bxh�Ŵ  �          AdQ�@�G���33�>ff�f�RC�8R@�G�>��W�
��?�\)                                    Bxh��Z  �          Aap�@'����\�N{ǮC�9�@'�?�Q��Y{B\)                                    Bxh��   �          AT��@?\)�%��I�#�C���@?\)@q��DQ�{BN��                                    Bxh��  
�          AM녿z�@�  �8(��HB��f�z�A
=����(�B��
                                    Bxh� L  T          AB�\����@W
=��RaHB�\)����@�\)��\�)�B���                                    Bxh��  �          A<zᾙ��@333�3��)B�\����@����
�<�RB�L�                                    Bxh��  T          A>ff���
@�ff�%��x=qB��)���
A����=q�G�B�(�                                    Bxh�,>  |          A9G���  A  ��(��	{B�8R��  A-p����
��B���                                    Bxh�:�  
�          A*ff?�G�@��
��p��#��B�� ?�G�@��Ϳ\�O�
B�k�                                    Bxh�I�  "          AHQ�@��
=���  C�� @��k��,���t�\C���                                    Bxh�X0  �          AE�@�(���=q�����C�S3@�(��333�)G��m(�C�˅                                    Bxh�f�  	�          AD��@��H��{�\)�J  C���@��H�8Q��.�\�{z�C�
                                    Bxh�u|  T          AN{@�����ff�*ff�dQ�C���@���?�ff�9p�A<z�                                    Bxh��"  
�          AT  @��H��=q�3��n=qC��@��H@u�+\)�]=qA��                                    Bxh���  
<          APQ�@��R�5�!p��jz�C�:�@��R@=q�#33�nA���                                    Bxh��n  |          AW�@��@�����;�BpG�@��A;
=��
=����B�\)                                    Bxh��  �          Ai@�ffA8����R���B���@�ffAY�.{�,(�B���                                    Bxh���  T          Au�@�(�A
=� Q��#��Bm��@�(�AX���z�H�mp�B��                                    Bxh��`  T          Aup�@�33A(  ������B{�@�33A_
=�\(��O�
B���                                    Bxh��  
�          As�@J�HAF�\� ��� ��B�Q�@J�HAl(��z�H�n{B��q                                    Bxh��  �          A}�@=qAQG��	���HB�#�@=qAy녿�33���B�z�                                    Bxh��R  "          A|(�@hQ�AEG������B�#�@hQ�Ar�H�����z�B�{                                    Bxh��  
�          Ayp�@�{A(z�� (�� Bz��@�{AaG��e�V{B�Ǯ                                    Bxh��  
Z          A�  @ϮA9G�����  Br�@ϮAjff����
�HB�33                                    Bxh�%D  
�          A�\)@�(�A6=q�
=��Br��@�(�Al���Fff�-�B�W
                                    Bxh�3�  
�          A�{@��RA���Ap��=�Bm{@��RAd�����H��Q�B�u�                                    Bxh�B�  
�          A�@��@�(��h  �x{B�Ǯ@��AX���Q��Q�B��                                    Bxh�Q6  h          A�@\)A ���\z��c��B~p�@\)A\Q�� �����HB���                                    Bxh�_�  "          A�z�@�@�
=�s�
�)B��@�A5��1p��/  B�B�                                    Bxh�n�  |          A�  @�
=A��X  �[�RBip�@�
=AZ�\��G��ᙚB�                                      Bxh�}(  �          A��@�RAQ��>�H�0��BR�\@�RAe����H��ffBzQ�                                    Bxh���  �          A���@��@���k���B%@��A/
=�+�
�'B��\                                    Bxh��t  
�          A���@��@��tz��|��B(\)@��A@z��,  ��B�                                      Bxh��  "          A��@�33@���~�H�RB {@�33AA��7
=�!ffB�u�                                    Bxh���  "          A��R@��@4z��xz��\Aͮ@��A!�A��833Bo��                                    Bxh��f  �          A�33@�(��:�H��(���C�"�@�(�@�Q��k��g�\BY��                                    Bxh��  
�          A�  @����5���
=C��q@���@�(��n{�i��Ba��                                    Bxh��  T          A�Q�@�(�?\)��(���@���@�(�AQ��]��V�Bb(�                                    Bxh��X            A���@��@��}p�ffA��@��A��K
=�;��B[��                                    Bxh� �  
�          A��@��
��{�f�H�gC�L�@��
@Q��k��o�RA��                                    Bxh��  T          A�G�@�=q�L�����
�C��@�=q@�Q��z�H�w�B��                                    Bxh�J  T          A��H@��H�z�H��z��zz�C���@��H@����~=q�u�A��\                                    Bxh�,�  T          A�z�@�  ���b=q�p  C�h�@�  @.�R�k���A��                                    Bxh�;�  f          A�
=@�Q�����c��G  C��=@�Q쿹�����)C�+�                                    Bxh�J<  
*          A��@�=q��G��333��{C�� @�=q�ff����f�HC�<)                                    Bxh�X�  �          A�{@љ������\)��C���@љ��0���f{�@��C�N                                    Bxh�g�            Ag33@��
�J�R�   �33C���@��
�   � z����C��=                                    Bxh�v.  |          A��RA�
�O\)���
��ffC���A�
�33�<  �3�C�H                                    Bxh���  �          AM�@N�R�"�R�������RC���@N�R����(��KC�                                      Bxh��z  �          @��H�G����H��Q�s33C���G����\�/\)���
C�/\                                    Bxh��   �          AC�
@�\)��
=�	p��3�C��3@�\)��{�(���p{C��H                                    Bxh���  �          A((�@�\)��G����\�A\)C��@�\)�c�
���L�C��                                    Bxh��l  �          A6ff@~{��33�p��W�C�G�@~{�����(��C�H�                                    Bxh��  �          Az�@G��@  ��ff�oQ�C�\@G�?�\)��ff�qA��
                                    Bxh�ܸ  |          A%��@8Q쿕��G��r�
C��R@8Q�?�G���(��g�A�
=                                    Bxh��^  T          A�>W
=Ap��?\)���B�\>W
=A�?˅A$��B�.                                    Bxh��  |          A?@  A�H��w33B��
?@  A�@(�Ag\)B��H                                    Bxh��  �          A�\>�(�@�(����H�G�B�W
>�(�A
=�����=qB���                                    Bxh�P  �          A  ?G�@�G������  B�ff?G�A녿�G���B���                                    Bxh�%�  �          A�R>W
=@�=q��  �D33B�>W
=AQ��AG����B�=q                                    Bxh�4�  �          A!p�>�Q�@E���{B���>�Q�@�R��\)�(�B�p�                                    Bxh�CB  �          A>�R?&ff@<���8Q���B�{?&ffAG����5�RB�\                                    Bxh�Q�  �          A>{?��?��9¡A�ff?��@�G��Q��`�HB�G�                                    Bxh�`�  �          A��?#�
�3�
�33ffC�l�?#�
?�Q���H.B��\                                    Bxh�o4  �          AG�?5�����ff�mffC�O\?5����«8RC���                                    Bxh�}�  �          A�>�p����
��33�L�C���>�p����H�ff¢8RC�(�                                    Bxh���  �          A  ?��H���
�陚�QG�C��R?��H�xQ��(�\)C�E                                    Bxh��&  �          A�H@���Q��ָRG�C���@�?ٙ������3B{                                    Bxh���  �          A��@녿8Q����G�C�<)@�@O\)��\)�q�RB[33                                    Bxh��r  �          A
=q@p  ?���߮�l��A�Q�@p  @�=q��(��z�BU��                                    Bxh��  �          A(�@��?5��\)�_33AG�@��@�Q���(��)=qB$�                                    Bxh�վ  �          Az�@�G�?�p���z��g��A�@�G�@�
=��  ��BJff                                    Bxh��d  �          A
�H@�(��33�ָR�[p�C�s3@�(�?�z���ff�g��A�ff                                    Bxh��
  �          Az�@��׿^�R���TQ�C�N@���@5�ҏ\�>�RA�z�                                    Bxh��  �          A��@��H�(���(��S
=C���@��H?�  ��(��^(�A�                                    Bxh�V  �          A��@�Q�������z��{C��)@�Q�� ��� (��iG�C�                                      Bxh��  �          A@QG���p��u����C�޸@QG����R��=q�O��C��                                    Bxh�-�  �          AG�@y����{�E����HC��
@y�������p��7ffC�b�                                    Bxh�<H  �          A�R@n�R���c�
��z�C�W
@n�R��ff�����Q�C��                                    Bxh�J�  �          A\)@����\)�O\)���C���@���(���{�4ffC��                                    Bxh�Y�  �          @�?z�H@�Q��(��i��B��
?z�H@�\)?���Axz�B�Ǯ                                    Bxh�h:  �          @���@�ff@1��S33�B��@�ff@�=q��
=�[�
B0G�                                    Bxh�v�  �          @�R@����{�p����C�@������.{�Σ�C��=                                    Bxh���  �          @�p�@Y����{@33A��C�\)@Y����?s33AMC�>�                                    Bxh��,  �          @�z�@U�}p�?�A�C��R@U��\)�u�p��C��q                                    Bxh���  �          @�p��녿��
�N{�|\)Cy����>��
�c3333C�f                                    Bxh��x  �          @�����33?�p��h����C#���33@G��Q���p�C33                                    Bxh��  �          @�{@N{���?xQ�A��C��@N{��=q� �����C��3                                    Bxh���  �          A��@��H��?��A�\C��q@��H��녿����O�C�E                                    Bxh��j  �          Az�@�����\?�A��C�<)@�������  �-�C�t{                                    Bxh��  �          A\)@�(����
@ffAf�RC��
@�(���=q�8Q����
C��q                                    Bxh���  �          A
=A   ��ff@��ALz�C��qA   ��녿c�
��z�C�H                                    Bxh�	\  T          A�@�ff�θR?�p�A8��C��@�ff��=q�������C���                                    Bxh�  �          A&ff@�  ��@(Q�Aj�HC�%@�  ���H����ʏ\C�L�                                    Bxh�&�  |          A!��A z���z�?�  A\)C�xRA z������=q��HC��                                    Bxh�5N  �          A"{A���K��8Q���33C���A�����  �M�C�                                    Bxh�C�  �          A&�RA�����\���Ϳ
=qC�q�A���]p��
�H�@��C�                                    Bxh�R�  �          A,z�A�R���\?E�@��
C�W
A�R���\����C��3                                    Bxh�a@  �          A$Q�A{��녿.{�r�\C�+�A{��  �J�H��{C��                                     Bxh�o�  �          A�HA
=q���
=�\)>\C�G�A
=q��
=�(��c�C��R                                    Bxh�~�  �          Ap�@�����R���N{C��H@���\)�>�R����C���                                    Bxh��2  �          @�@�����G��aG����HC��q@����r�\� �����C�{                                    Bxh���  �          A�@�  ��33�L�;��RC�%@�  ��G��:=q��=qC�Ff                                    Bxh��~  
�          A{@��
��
=�   �Dz�C�{@��
����W
=����C��R                                    Bxh��$  �          A�
@����{=u>�p�C�.@�����R�/\)��33C�'�                                    Bxh���  �          A�@�Q����ÿ�����\)C�\@�Q�������  ��  C��f                                    Bxh��p  |          A!�A�
�   @p�Ac�C�\A�
�>�R?��@ǮC�L�                                    Bxh��  �          A,  Ap���ff@U�A��C��{Ap�����?�@E�C�Y�                                    Bxh��  �          A-G�A�\��
=@(�A<  C��\A�\��p��
=q�8��C���                                    Bxh�b  �          A.�\A���33?z�H@�C�� A������33�Q�C��                                    Bxh�  �          A7
=A&�\��z�?Y��@���C��qA&�\��p���33���C��                                     Bxh��  �          A<z�A+
=��(�?Q�@~{C���A+
=��33��ff���C�Z�                                    Bxh�.T  �          A>=qA/\)��{?O\)@|(�C��)A/\)��
=��=q���C�w
                                    Bxh�<�  �          A{A��Fff�#�
�#�
C��A��*=q�����C��{                                    Bxh�K�  �          A#�A���0�׾��:�HC��A���33���
�
�\C�aH                                    Bxh�ZF  �          A ��Ap��333>�ff@%C��qAp��(Q쿆ff��G�C�Q�                                    Bxh�h�  
�          A-��A ���^{?�AffC�y�A ���z=q�u��  C�aH                                    Bxh�w�  �          A+33AG����
�����z�C��fAG���G��hQ����C�1�                                    Bxh��8  �          A*ffA�\��?xQ�@�\)C�� A�\���
��6{C�xR                                    Bxh���  �          A((�Ap�����?�=qA ��C�{Ap��Å����\)C��f                                    Bxh���  �          A2�\AG��z�H@dz�A�G�C�"�AG���ff?�ff@��
C�\                                    Bxh��*  �          A0��Aff�
=@�
=A�ffC���Aff���@/\)AeG�C���                                    Bxh���  �          A-p�A����  @:�HA~ffC��{A����  �W
=��{C��
                                    Bxh��v  �          A��@�{���H@���A�33C���@�{����?�{@�\)C��                                    Bxh��  �          A$��@�
=@��@�  B$z�B8Q�@�
=?���A (�Be��AX��                                    Bxh���  �          A��@���?��
@�G�BB�A�33@����\)@�p�B>p�C��                                    Bxh��h  |          A��@��?��@��B
\)A2�H@�녿�G�@�  B
��C���                                    Bxh�
  �          A&�\A�H����@^�RA���C��3A�H�>�R@G�AJ�\C�s3                                    Bxh��  �          A+�A��p��@�A���C�B�A��=p�@EA�=qC���                                    Bxh�'Z  
�          A/�A  ����@���A�ffC���A  �Dz�@��A���C�
=                                    Bxh�6   �          A(z�A(���=q@�Q�B�C���A(����@��RA���C��                                     Bxh�D�  �          A0Q�A(�?�Q�@�33B!p�A33A(��)��@�{B�\C��                                    Bxh�SL  �          A�R@{@��@�{B2ffB�@{@�A��B�(�B)�R                                    Bxh�a�  �          A&�R��\)A\)@�\)B{B��f��\)@���A�
B�ǮB���                                    Bxh�p�  �          A(Q�?��\A�H@�
=A��\B�?��\@�=qA��Bx\)B�33                                    Bxh�>  �          A(z�?h��A
=@�  A��B�\)?h��@��AG�ByG�B��=                                    Bxh���  �          A$��@g
=@���@љ�BBG�BM��@g
=>�@�{B���@�\                                    Bxh���  �          A0  ��R@��@�
=A���Bٮ��R@q�@���Bo��B��=                                    Bxh��0  �          A.ff���\A��@�Q�A���B�����\@�=q@�Q�B?��C�)                                    Bxh���  �          A
=?u@��H@��BD  B�W
?u?�(�Ap�B��Bc                                    Bxh��|  �          A��@�  @.{@�\BP�A��@�  ���H@�B_p�C���                                    Bxh��"            A9G�Az�����@�33A��C�� Az����@�
A((�C�{                                    Bxh���  �          AD  @����33@��RAə�C�l�@����\>Ǯ?�C��
                                    Bxh��n  �          A<(�@�ff���@�z�A��\C�e@�ff��׿G��s�
C��f                                    Bxh�  �          A;\)@�ff����@�z�A��C�Y�@�ff��;��H���C��R                                    Bxh��  �          AD��@�  �!G�@EAiG�C��f@�  �#��%��B�RC���                                    Bxh� `  �          AF=q@�{�1?��@�z�C���@�{�$����{���\C�]q                                    Bxh�/  �          A@��@�{��ff@�(�A��C���@�{�\)>�=q?��C�R                                    Bxh�=�  h          AEG�@�Q���(�@�(�A��C�T{@�Q��Q�=�Q�>�G�C�
=                                    Bxh�LR  �          AR=q@.�R�K�
?��@�\)C�@.�R�9������(�C��                                    Bxh�Z�  �          ANff@(��E@ffA(��C���@(��<����������C�ٚ                                    Bxh�i�  �          ADz�@9���=�?�A33C��H@9���1���������C�O\                                    Bxh�xD  �          A5G�A�\�K�@�B��C���A�\���@���A�G�C��                                    Bxh���  �          A9A�R�R�\@�  B��C�Q�A�R�ҏ\@��A�=qC��                                    Bxh���  ,          AM�A����@�{Bz�C�A���=q@.�RAH  C��\                                    Bxh��6  �          AR=q@��H�"�R@Z�HAt(�C��@��H�'��ff�&�RC���                                    Bxh���  �          AN=q@�
=�0  @��A"=qC�7
@�
=�(���u�����C��{                                    Bxh���  �          AM@���2�R@I��Ad(�C���@���2�H�Fff�`(�C���                                    Bxh��(            AL(�@�Q��(  @r�\A��C��@�Q��/
=�p�� ��C�c�                                    Bxh���  T          AIp�@�G��*�R@N�RAn�\C�<)@�G��,���1��L(�C�
                                    Bxh��t  T          AJ=q@�33�+33@P  Ao33C�U�@�33�-G��1G��J�HC�/\                                    Bxh��  �          AT(�@�{�5G�@L(�A_�
C��
@�{�5p��J=q�]��C���                                    Bxh�
�  "          AR�H@�ff�:�R@*=qA<Q�C��@�ff�5�q�����C�N                                    Bxh�f  �          AQ��@��
�z�@��A�p�C���@��
�-G�>�(�?�{C�q                                    Bxh�(  
�          AZ�\@�(��?\)@HQ�AS�
C��@�(��=p��a��o
=C�4{                                    Bxh�6�  "          AH��@�{�\)@��RA���C���@�{�!�>�=q?��RC�:�                                    Bxh�EX  �          AM@�����@�33A�33C�g�@����*�\�aG��}p�C�u�                                    Bxh�S�  
�          AIG�@�G���@�Q�A�  C���@�G��*�\�������HC���                                    Bxh�b�  
n          AMp�@أ���\@�=qA�{C�j=@أ��33>8Q�?^�RC�)                                    Bxh�qJ  �          AH��@������@�p�A�C���@����*{��녿��C��=                                    Bxh��  
�          AH(�@���33@���A�C�  @��?G�@hQ�C�#�                                    Bxh���  "          AMG�A �����@�
=A�p�C�>�A ����R>���?�C��R                                    Bxh��<  �          AR�\A
=@У�@��A��HB
�\A
=@#33A��B"  Ar=q                                    Bxh���  
�          AS�A@��
@�BQ�A�(�A>�33A�B&z�@Q�                                    Bxh���  	�          AMA!G��p�@��BffC���A!G���33@���A�C�f                                    Bxh��.  	�          AK�
@�
=��@p�A.�HC��H@�
=�G��E�v�\C�ٚ                                    Bxh���            AQ��@陚����@�z�A�ffC��3@陚��{<��
=��
C�N                                    Bxh��z  T          AS
=A<��?�p�@��
A��A��A<�Ϳ���@��
A��C�W
                                    Bxh��   @          APz�A9?��@���A�z�A z�A9����@�(�AŮC��                                    Bxi �  
�          AV�\@�G�����@�{B!G�C��q@�G�����@qG�A��RC�G�                                    Bxi l  �          AYp�A1��<��@��HA�Q�C��{A1���{@�33A�  C���                                    Bxi !  
�          AZ=qA8Q�����@�33A���C�5�A8Q���{?���@���C��\                                    Bxi /�  @          AV{@��"�\?�\@�C��@�����p����{C�U�                                    Bxi >^  "          AUG�@�=q�?33?��\@�(�C��)@�=q�-����
��z�C��{                                    Bxi M  

          AV�\@أ��/�@FffAXz�C���@أ��/�
�C33�T��C���                                    Bxi [�  
�          AU��@`  �B�R�n{��C��@`  ��R���=qC���                                    Bxi jP  T          AW�@���8(��\)�-C�� @�������ffC��                                    Bxi x�  �          A_33@��H�@Q��(��33C�xR@��H�  ���
=C��3                                    Bxi ��  �          A^�R@�\)��R�z����C�"�@�\)���C��}�\C�xR                                    Bxi �B  T          AX��@c33�,(����H�C�  @c33��(��=�u�HC���                                    Bxi ��  
�          AU@˅�Q�������C�7
@˅�����-��ZC��                                    Bxi ��  
�          AY��@����	�?�
=@�  C�W
@������N�R�33C�4{                                    Bxi �4            An=qA"�R�{@�p�A���C��A"�R�"ff?��@�ffC��f                                    Bxi ��  
(          Ar�HA)���\A�HB�
C�5�A)��
@�p�A��\C�|)                                    Bxi ߀            AnffA ���C33?=p�@:=qC���A ���,����ff��G�C�Z�                                    Bxi �&  r          Al  @���J�R��ff��(�C�}q@���"ff��{��\)C�S3                                    Bxi ��  �          Ab=q@�(��9��?�\)@�(�C��=@�(��'�
�����z�C�H                                    Bxir  �          A[�A���@�
=A���C�3A��+����Ϳ�
=C�.                                    Bxi  "          AX  A
�R�
=@^�RAo�C��=A
�R�!p���
�z�C�XR                                    Bxi(�  �          AS�A ���$Q�?�@�z�C�\A �����q���p�C���                                    Bxi7d  T          AR�\@�z��+
=?�
=A
=C�T{@�z��"=q�y����{C��                                    BxiF
  
�          AT(�A
=�$  ?��A�HC�Q�A
=��
�mp����
C��                                    BxiT�  �          A\Q�@�ff�$��@�ffA�
=C��
@�ff�/���  ���
C��{                                    BxicV  
Z          AV{@������@�{A��C��@����3\)��G���C��                                    Bxiq�  
Z          A\��@Ϯ�2�R��z���
=C��@Ϯ����  �Q�C�.                                    Bxi��  "          AlQ�@���5p�@�(�A�z�C���@���P  ��(���z�C�0�                                    Bxi�H  "          Ao�@���8(�@�{A�ffC��3@���P�Ϳ+��%�C�aH                                    Bxi��  T          Ao\)@����G33@�\)A�\)C���@����Q���=q�{C�                                    Bxi��  "          Arff@ڏ\�C
=@�A��C�� @ڏ\�X  ��  ��
=C�j=                                    Bxi�:  �          Ap��@�G��HQ�@��A�C�g�@�G��\�ÿ�\)���RC�]q                                    Bxi��  
�          At��@�\)�J{@�Q�A���C�  @�\)�^�R��z�����C�                                    Bxi؆  T          Aq��@�R�B=q@�{A���C��f@�R�O
=�z����
C��R                                    Bxi�,  !          Ah(�@���R�\@.�RA.{C��)@���J�R��\)����C�]q                                    Bxi��  �          Af�H@�Q��D��?�ff@�ffC�J=@�Q��733��=q���RC�1�                                    Bxix  "          Ac33@�\)�H  =���>ǮC�%@�\)�+�
��z���{C�f                                    Bxi  
�          AA�@W���A"ffBn�C�*=@W���@�(�A���C���                                    Bxi!�  "          AI@���A{B'p�C��@��A��@{A5C�S3                                    Bxi0j  	�          AL(�@%��2ff@�ffAӮC�� @%��G�
�fff����C��3                                    Bxi?  T          AD��>�z���@��B(��C��>�z��:�H@G�A0(�C��3                                    BxiM�  
�          A5>�=q�!�@�\)A�Q�C��f>�=q�1����Q��ÅC���                                    Bxi\\  T          A���R��?��R@�G�C�R��R�	��p����Q�C��                                    Bxik  �          A   �/\)�  �0������C}�f�/\)������Q��  Cz�\                                    Bxiy�  �          A7\)�����%G���p��{Cx��������ff�ڏ\��RCs�                                    Bxi�N  �          A!���ff�����  �2(�CG����ff?�=q��\)�9�C&�                                    Bxi��  �          A2{��=q��  � (��:p�C5޸��=q@{��߮�G�C��                                    Bxi��  �          A3
=��G�=u���R�6z�C3����G�@�������{C�)                                    